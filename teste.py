#!/usr/bin/env python
# coding: utf-8

import sys
import logging
import multiprocessing
import gc  # <--- para coleta de lixo manual

# PyTorch e Lightning
import torch
from torch import nn
import lightning.pytorch as L
from lightning.pytorch import Trainer

# Ray (tune/grid search)
from ray import tune, put, get, init, is_initialized
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler
from functools import partial

# Plot
import matplotlib.pyplot as plt
import seaborn as sns

# MLflow (opcional)
import mlflow

# NumPy, Pandas, sklearn
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split

########################
# LOGGING
########################
logging.basicConfig(
    level=logging.INFO,
    filename="training_logs.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

########################
# 1. LEITURA DE DADOS
########################
df = pd.read_parquet("data.parquet")
df = df.astype("float32")

target_column = "perf"
y = df[target_column]
X = df.drop(columns=[target_column])

del df
gc.collect()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

del X, y
gc.collect()

X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

del X_train, X_test, y_train, y_test
gc.collect()

########################
# 2. Preparando Ray (Object Store)
########################
X_train_ref = put(X_train_tensor)
X_test_ref = put(X_test_tensor)
y_train_ref = put(y_train_tensor)
y_test_ref = put(y_test_tensor)

# Podemos remover os tensores locais se quisermos
del X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor
gc.collect()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device detectado: {device}")

########################
# 3. Datasets e Dataloaders
########################
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Recuperamos os dados do Ray Object Store
train_dataset = CustomDataset(get(X_train_ref), get(y_train_ref))
test_dataset = CustomDataset(get(X_test_ref), get(y_test_ref))

num_workers = max(1, min(8, multiprocessing.cpu_count() // 2))

train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=512,
    shuffle=True,
    num_workers=num_workers
)
test_dataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=512,
    shuffle=False,
    num_workers=num_workers
)

########################
# 4. Definição do Modelo (Lightning)
########################
class LightningModel(L.LightningModule):
    def __init__(self, num_features, activation="ReLU", optimizer_name="Adam", loss_name="MSE"):
        super().__init__()
        self.num_features = num_features
        self.activation = activation
        self.optimizer_name = optimizer_name
        self.loss_name = loss_name
        self.model = self.build_model()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.get_loss_function()(preds, y.unsqueeze(1))
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.get_loss_function()(preds, y.unsqueeze(1))
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return self.get_optimizer()

    def get_optimizer(self):
        opts = {
            "Adam": torch.optim.Adam(self.parameters(), lr=1e-3),
            "AdamW": torch.optim.AdamW(self.parameters(), lr=1e-3),
        }
        return opts[self.optimizer_name]

    def get_loss_function(self):
        losses = {
            "MSE": nn.MSELoss(),
            "SmoothL1Loss": nn.SmoothL1Loss(),
            "MAE": nn.L1Loss()
        }
        return losses[self.loss_name]

    def get_activation(self):
        acts = {
            "ReLU": nn.ReLU(),
            "PReLU": nn.PReLU(),
            "ELU": nn.ELU()
        }
        return acts[self.activation]

    def build_model(self):
        # Exemplo: duas camadas ocultas
        hidden_size = self.num_features // 2
        hidden_size2 = hidden_size // 2

        return nn.Sequential(
            nn.Linear(self.num_features, hidden_size),
            self.get_activation(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size2),
            self.get_activation(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size2, 1)
        )

########################
# 5. Função de treino
########################
def train_model_tune(config, num_features, train_dataloader, val_dataloader, max_epochs=20):
    torch.set_float32_matmul_precision("medium")

    model = LightningModel(
        num_features=num_features,
        activation=config["activation"],
        optimizer_name=config["optimizer"],
        loss_name=config["loss_function"]
    )

    is_headless = not sys.stdout.isatty()

    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices="auto",
        enable_progress_bar=not is_headless
    )
    trainer.fit(model, train_dataloader, val_dataloader)

########################
# 6. Função de tuning
########################
def tune_hyperparameters(num_features, train_dataloader_ref, val_dataloader_ref, num_samples=10):
    config = {
        "optimizer": tune.choice(["Adam", "AdamW"]),
        "loss_function": tune.choice(["MAE", "MSE", "SmoothL1Loss"]),
        "activation": tune.choice(["ReLU", "PReLU", "ELU"])
    }

    search_algo = OptunaSearch(metric="loss", mode="min")
    scheduler = ASHAScheduler(
        time_attr="training_iteration",
        max_t=30,
        grace_period=10,
        reduction_factor=2
    )

    tuner = tune.Tuner(
        tune.with_resources(
            partial(
                train_model_tune,
                num_features=num_features,
                train_dataloader=get(train_dataloader_ref),
                val_dataloader=get(val_dataloader_ref),
                max_epochs=20
            ),
            resources={"cpu": 10, "gpu": 1}
        ),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            search_alg=search_algo,
            scheduler=scheduler,
            num_samples=num_samples
        ),
        param_space=config
    )

    results = tuner.fit()

    # Pegar o melhor trial
    best_result = results.get_best_result(metric="loss", mode="min")
    best_trial_config = best_result.config
    best_trial_loss = best_result.metrics["loss"]

    print(f"Best trial config: {best_trial_config}")
    print(f"Best trial final validation loss: {best_trial_loss}")
    return best_trial_config

########################
# 7. Execução principal
########################
if __name__ == "__main__":
    
    if not is_initialized():
        init()
    
    gc.collect()
    
    num_features = train_dataset.features.shape[1]

    best_config = tune_hyperparameters(
        num_features=num_features,
        train_dataloader_ref=put(train_dataloader),
        val_dataloader_ref=put(test_dataloader),
        num_samples=18
    )
    print("Melhor config encontrada:", best_config)
