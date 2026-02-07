# This script prunes and quantizes HEPT using activation pruning and fixed-point quantization.
# It saves the best compressed model (checks every epoch) based on validation accuracy.
# Change the PATH variable to set where to save the model and logs.
# If you want to resume from a previous model, set the RESUME_MODEL_NAME variable accordingly.

import nni
import yaml
import argparse
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from datetime import datetime
import numpy as np
import torch
from torchmetrics import MeanMetric
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR, StepLR
from torch_geometric.utils import unbatch, remove_self_loops, to_undirected, subgraph
from torch_geometric.nn import knn_graph, radius_graph
from utils import set_seed, get_optimizer, log, get_lr_scheduler, add_random_edge, get_loss
from utils.get_data import get_data_loader, get_dataset
from utils.get_model import get_model
from utils.metrics import acc_and_pr_at_k, point_filter
import os
os.environ["KERAS_BACKEND"] = "torch"  # Some pruning layers / quantizers expect Keras backend env var
import torch.nn.functional as F
from pquant import (
    get_default_config,
    add_compression_layers,
    iterative_train,
    get_layer_keep_ratio,
    get_model_losses,
    remove_pruning_from_model,
)
from quantizers.fixed_point.fixed_point_ops import get_fixed_quantizer

import json
PATH = "/home/divij/HEPT/data/tracking/logs/mlp_32_wanda_0.6/" # Change this
PQUANT_RESULTS = "pquant_results.txt"
RESUME_MODEL_NAME = "best_model.pt" #"compressed_model.pth"

excluded_from_pruning = [
    "feat_encoder.0",
    "feat_encoder.2",
    "W",
    "mlp_out.0",
    "mlp_out.12"
]

def save_config(config, filename = "pquant_config.txt"):
    print(f"OUTPUTTING CONFIG SETTINGS TO {PATH + filename}")
    with open(PATH + filename, "w") as f:
        json.dump(config, f, indent=4)

def enable_quantization(config, i, f, hgq=True):
    config["quantization_parameters"]["enable_quantization"] = True
    config["quantization_parameters"]["default_fractional_bits"] = f
    config["quantization_parameters"]["default_integer_bits"] = i
    config["quantization_parameters"]["use_symmetric_quantization"] = True
    config["quantization_parameters"]["use_high_granularity_quantization"] = hgq

    return config

def prune_with_wanda(excluded_layers):
    pretraining_rounds = 1
    config = get_default_config("wanda")
    config["pruning_parameters"]["enable_pruning"] = True
    config["pruning_parameters"]["t_delta"] = 5
    config["pruning_parameters"]["t_start_collecting_batch"] = 5
    config["pruning_parameters"]["sparsity"] = 0.6
    config["pruning_parameters"]["disable_pruning_for_layers"] = excluded_layers
    # config["pruning_parameters"]["N"] = 2
    # config["pruning_parameters"]["M"] = 4

    config["training_parameters"]["pretraining_epochs"] = pretraining_rounds
    config["training_parameters"]["epochs"] = 20

    return config

def prune_with_pdp(excluded_layers):
    pretraining_rounds = 1
    config = get_default_config("pdp")
    pruning_cfg = config["pruning_parameters"]
    pruning_cfg["enable_pruning"] = True
    # pruning_cfg["sparsity"] = 0.8
    # pruning_cfg["epsilon"] = 0.05
    # pruning_cfg["temperature"] = 1e-5
    # pruning_cfg["structured_pruning"] = False
    pruning_cfg["disable_pruning_for_layers"] = excluded_layers

    training_cfg = config["training_parameters"]
    training_cfg["pretraining_epochs"] = 10
    # training_cfg["fine_tuning_epochs"] = 0
    # training_cfg["epochs"] = 10
    # training_cfg["pruning_first"] = False
    # training_cfg["rewind"] = "never"
    # training_cfg["rounds"] = 1
    return config

def prune_with_activation_pruning(excluded_layers):
    # pruning_methods: "autosparse, cl, cs, dst, pdp, wanda"
    pruning_method = "ap"
    config = get_default_config(pruning_method)
    # Set target sparsity to 80% (20% of weights are non-zero). This parameter exists only for some pruning methods
    # config["pruning_parameters"]["sparsity"] = 0.8
    # save_config(config, filename = "pquant_config.txt")

    config["pruning_parameters"]["enable_pruning"] = True
    config["training_parameters"]["epochs"] = 100
    config["training_parameters"]["pretraining_epochs"] = 10
    config["pruning_parameters"]["threshold"] = 0.3

    config["pruning_parameters"]["disable_pruning_for_layers"] = excluded_layers

    return config

def total_weights(model):
    total = sum(p.numel() for name, p in model.named_parameters() if "weight" in name)
    return total

def weights_for_layers(model, layer_names):
    layer_names = set(layer_names)  # faster lookup
    total = 0
    for name, param in model.named_parameters():
        # Check if the parameter belongs to any of the specified layers
        if "weight" in name:
            prefix = ".".join(name.split(".")[:-1])  # remove .weight or .bias
            if prefix in layer_names:
                total += param.numel()
    return total


def train_one_batch(model, optimizer, criterion, data, lr_s):
    model.train()
    embeddings = model(data)
    loss = criterion(embeddings, data.point_pairs_index, data.particle_id, data.reconstructable, data.pt)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if lr_s is not None and isinstance(lr_s, LambdaLR):
        lr_s.step()
    return loss.item(), embeddings.detach(), data.particle_id.detach()


@torch.no_grad()
def eval_one_batch(model, optimizer, criterion, data, lr_s):
    model.eval()
    embeddings = model(data)
    loss = criterion(embeddings, data.point_pairs_index, data.particle_id, data.reconstructable, data.pt)
    return loss.item(), embeddings.detach(), data.particle_id.detach()


def process_data(data, phase, device, epoch, p=0.2):
    data = data.to(device)

    if phase == "train":
        # pairs_to_add = add_random_edge(data.point_pairs_index, p=p, batch=data.batch, force_undirected=True)
        num_aug_pairs = int(data.point_pairs_index.size(1) * p / 2)
        pairs_to_add = to_undirected(torch.randint(0, data.num_nodes, (2, num_aug_pairs), device=device))
        data.point_pairs_index = torch.cat([data.point_pairs_index, pairs_to_add], dim=1)

    return data


def run_one_epoch(model, optimizer, criterion, data_loader, phase, epoch, device, metrics, lr_s):
    run_one_batch = train_one_batch if phase == "train" else eval_one_batch
    phase = "test " if phase == "test" else phase

    # Just for progress bar
    pbar = tqdm(data_loader, disable=__name__ != "__main__")

    # idx = index of the batch, data is the batch data
    for idx, data in enumerate(pbar):
        if phase == "train" and model.attn_type == "None":
            torch.cuda.empty_cache()
        data = process_data(data, phase, device, epoch)

        batch_loss, batch_embeddings, batch_cluster_ids = run_one_batch(model, optimizer, criterion, data, lr_s)
        batch_acc = update_metrics(metrics, data, batch_embeddings, batch_cluster_ids, criterion.dist_metric)
        metrics["loss"].update(batch_loss)

        desc = f"[Epoch {epoch}] {phase}, loss: {batch_loss:.4f}, acc: {batch_acc:.4f}"
        if idx == len(data_loader) - 1:
            metric_res = compute_metrics(metrics)
            loss, acc = (metric_res["loss"], metric_res["accuracy@0.9"])
            prec, recall = metric_res["precision@0.9"], metric_res["recall@0.9"]
            desc = f"[Epoch {epoch}] {phase}, loss: {loss:.4f}, acc: {acc:.4f}, prec: {prec:.4f}, recall: {recall:.4f}"
            reset_metrics(metrics)
        pbar.set_description(desc)
    return metric_res


def reset_metrics(metrics):
    for metric in metrics.values():
        if isinstance(metric, MeanMetric):
            metric.reset()


def compute_metrics(metrics):
    return {
        f"{name}@{pt}": metrics[f"{name}@{pt}"].compute().item()
        for name in ["accuracy", "precision", "recall"]
        for pt in metrics["pt_thres"]
    } | {"loss": metrics["loss"].compute().item()}


def update_metrics(metrics, data, batch_embeddings, batch_cluster_ids, dist_metric):
    embeddings = unbatch(batch_embeddings, data.batch)
    cluster_ids = unbatch(batch_cluster_ids, data.batch)

    for pt in metrics["pt_thres"]:
        batch_mask = point_filter(batch_cluster_ids, data.reconstructable, data.pt, pt_thres=pt)
        mask = unbatch(batch_mask, data.batch)

        res = [acc_and_pr_at_k(embeddings[i], cluster_ids[i], mask[i], dist_metric) for i in range(len(embeddings))]
        res = torch.tensor(res)
        metrics[f"accuracy@{pt}"].update(res[:, 0])
        metrics[f"precision@{pt}"].update(res[:, 1])
        metrics[f"recall@{pt}"].update(res[:, 2])
        if pt == 0.9:
            acc_09 = res[:, 0].mean().item()
    return acc_09

def train_for_pquant(model, trainloader, device, loss_func, epoch, optimizer, scheduler, *args, **kwargs):
    metrics = kwargs["metrics"]
    config = kwargs["model_config"]
    lr_s = scheduler
    main_metric = config["main_metric"]

    train_res = run_one_epoch(model, optimizer, loss_func, trainloader, "train", epoch, device, metrics, lr_s)

    with open(PATH + PQUANT_RESULTS, "a") as f:
        f.write(
            f"[Epoch {epoch}] train: {train_res[main_metric]:.4f}, "
        )


def validate_and_test_for_pquant(model, testloader, device, loss_func, epoch, *args, **kwargs):
    lr_s = kwargs["scheduler"]
    metrics = kwargs["metrics"]
    config = kwargs["model_config"]
    pquant_config = kwargs["pquant_config"]
    log_dir = kwargs["log_dir"]
    best_valid = kwargs["best_valid"]
    opt = kwargs["opt"]

    main_metric = config["main_metric"]
    coef = 1 if config["mode"] == "max" else -1

    loaders = testloader
    criterion = loss_func

    valid_res = run_one_epoch(model, opt, criterion, loaders["valid"], "valid", epoch, device, metrics, lr_s)
    test_res = run_one_epoch(model, opt, criterion, loaders["test"], "test", epoch, device, metrics, lr_s)

    total_model_weights = total_weights(model)
    excluded_weights_count = weights_for_layers(model, excluded_from_pruning)

    ratio = get_layer_keep_ratio(model)
    ratio = (excluded_weights_count + (ratio * (total_model_weights - excluded_weights_count))) / total_model_weights

    with open(PATH + PQUANT_RESULTS, "a") as f:
        f.write(
            f"\nremaining_weights: {ratio * 100:.2f}%\n"
        )
    print(f"\nremaining_weights: {ratio * 100:.2f}%\n")

    if lr_s is not None:
        if isinstance(lr_s, ReduceLROnPlateau):
            lr_s.step(valid_res[config["lr_scheduler_metric"]])
        elif isinstance(lr_s, StepLR):
            lr_s.step()

    if ((valid_res[main_metric] * coef) > (best_valid[main_metric] * coef) and (int(ratio) != 1)):
        best_valid.update(valid_res)
        #torch.save(model.state_dict(), log_dir / "best_model.pt")
        compressed_model = remove_pruning_from_model(model, pquant_config)
        compressed_model.to(device)
        torch.save(compressed_model.state_dict(), PATH + "compressed_model.pth")
        print(f"New best compressed model found, saved at: \n{PATH + 'compressed_model.pth'}")

        with open(PATH + "best_model_stats.txt", "w") as f:
            f.write(
                f"remaining_weights: {ratio * 100:.2f}%\n"
                f"valid: {valid_res[main_metric]:.4f}, test: {test_res[main_metric]:.4f}\n\n"
            )

            f.write(
                # f"remaining_weights: {ratio * 100:.2f}%\n"
                f"Integer bits: {pquant_config['quantization_parameters']['default_integer_bits']} | "
                f"Fractional bits: {pquant_config['quantization_parameters']['default_fractional_bits']}\n\n"
            )

            for name, param in compressed_model.named_parameters():
                if param.requires_grad:
                    num_params = param.numel()
                    num_zeros = (param.cpu() == 0).sum().item()
                    percent = 100 * num_zeros / num_params
                    f.write(
                        f"{name}: {num_zeros} / {num_params} zero weights ({percent:.2f}%)\n"
                    )


    with open(PATH + PQUANT_RESULTS, "a") as f:
        f.write(
            f"valid: {valid_res[main_metric]:.4f}, test: {test_res[main_metric]:.4f}\n"
            f"\n\n"
        )


def run_one_seed(config):
    # Set device, number of threads
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(config["num_threads"])

    # logs details abt model
    dataset_name = config["dataset_name"]
    model_name = config["model_name"]
    dataset_dir = Path(config["data_dir"]) / dataset_name.split("-")[0]
    log(f"Device: {device}, Model: {model_name}, Dataset: {dataset_name}, Note: {config['note']}")

    # Create new dir to save model (will be saved later in this code)
    time = datetime.now().strftime("%m_%d-%H_%M_%S.%f")[:-4]
    rand_num = np.random.randint(10, 100)
    log_dir = dataset_dir / "logs" / f"{time}{rand_num}_{model_name}_{config['seed']}_{config['note']}"
    log(f"Log dir: {log_dir}")
    log_dir.mkdir(parents=True, exist_ok=False)
    writer = SummaryWriter(log_dir) if config["log_tensorboard"] else None

    set_seed(config["seed"])
    dataset = get_dataset(dataset_name, dataset_dir)
    loaders = get_data_loader(dataset, dataset.idx_split, batch_size=config["batch_size"])

    model = get_model(model_name, config["model_kwargs"], dataset)

    pquant_config = prune_with_wanda(excluded_from_pruning)
    input_shape = (1, 15)
    # pquant_config = enable_quantization(pquant_config, i=5, f=5, hgq=True)

    pquant_config["pruning_parameters"]["enable_pruning"] = True
    pquant_config["quantization_parameters"]["enable_quantization"] = False

    save_config(pquant_config, filename = "pquant_config.txt")
    model = add_compression_layers(model, pquant_config, input_shape)

    if config.get("only_flops", False):
        raise RuntimeError
    if config.get("resume", False):
        log(f"Resume from {config['resume']}")
        model_path = dataset_dir / "logs" / (config["resume"] + f"/{RESUME_MODEL_NAME}")
        checkpoint = torch.load(model_path, map_location="cpu")
        model.load_state_dict(checkpoint, strict=True)
    model = model.to(device) 

    opt = get_optimizer(model.parameters(), config["optimizer_name"], config["optimizer_kwargs"])
    config["lr_scheduler_kwargs"]["num_training_steps"] = config["num_epochs"] * len(loaders["train"])
    lr_s = get_lr_scheduler(opt, config["lr_scheduler_name"], config["lr_scheduler_kwargs"])
    criterion = get_loss(config["loss_name"], config["loss_kwargs"])

    main_metric = config["main_metric"]
    pt_thres = [0, 0.5, 0.9]
    metric_names = ["accuracy", "precision", "recall"]
    metrics = {f"{name}@{pt}": MeanMetric(nan_strategy="error") for name in metric_names for pt in pt_thres}
    metrics["loss"] = MeanMetric(nan_strategy="error")
    metrics["pt_thres"] = pt_thres

    coef = 1 if config["mode"] == "max" else -1
    # initialising the values (-inf for max, inf for min)
    best_epoch, best_train = 0, {metric: -coef * float("inf") for metric in metrics.keys()}
    best_valid, best_test = deepcopy(best_train), deepcopy(best_train)

    if writer is not None:
        layout = {
            "Gap": {
                "loss": ["Multiline", ["train/loss", "valid/loss", "test/loss"]],
                "acc@0.9": ["Multiline", ["train/accuracy@0.9", "valid/accuracy@0.9", "test/accuracy@0.9"]],
            }
        }
        writer.add_custom_scalars(layout)

    trained_model = iterative_train(model = model, 
                                config = pquant_config, 
                                train_func = train_for_pquant,
                                valid_func = validate_and_test_for_pquant, 
                                trainloader = loaders["train"], 
                                testloader = loaders, 
                                device = device, 
                                loss_func = criterion,
                                optimizer = opt, 
                                scheduler = lr_s,
                                metrics = metrics,
                                model_config = config,
                                log_dir = log_dir,
                                best_valid = best_valid,
                                opt = opt,
                                pquant_config=pquant_config
                                )

def main():
    parser = argparse.ArgumentParser(description="Train a model for tracking.")
    parser.add_argument("-m", "--model", type=str, default="hept")
    args = parser.parse_args()

    if args.model in ["gcn", "gatedgnn", "dgcnn", "gravnet"]:
        config_dir = Path(f"./configs/tracking/tracking_gnn_{args.model}.yaml")
    else:
        config_dir = Path(f"./configs/tracking/tracking_trans_{args.model}.yaml")
    config = yaml.safe_load(config_dir.open("r").read())
    run_one_seed(config)


if __name__ == "__main__":
    main()