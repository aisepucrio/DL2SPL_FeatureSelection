#!/usr/bin/env python
# coding: utf-8
import torch
from torch import nn
import lightning as L
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from ray import train, tune, put, get, init
from ray.tune.search.optuna import OptunaSearch
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
from ray.tune.schedulers import ASHAScheduler
from functools import partial
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


df = pd.read_parquet('data.parquet')
df = df.astype('float32')
df = df.sample(1500) # 1+ comente ou remova essa celula para rodar com o dataset completo, o tempo de execução será extenso.


num_features = df.shape[1]
num_features
target_column = 'perf'


y = df[target_column]
X = df.drop(columns=[target_column])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


X_train_np = X_train.to_numpy(dtype=np.float32)
X_test_np = X_test.to_numpy(dtype=np.float32)
y_train_np = y_train.to_numpy(dtype=np.float32)
y_test_np = y_test.to_numpy(dtype=np.float32)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)


init()

# Put tensors into the Ray object store
X_train_ref = put(X_train_tensor)
y_train_ref = put(y_train_tensor)
X_test_ref = put(X_test_tensor)
y_test_ref = put(y_test_tensor)


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Create datasets using references
train_dataset = CustomDataset(X_train_tensor, y_train_tensor)
test_dataset = CustomDataset(X_test_tensor, y_test_tensor)

# DataLoaders remain the same
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1024, shuffle=True, num_workers=23)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1024, shuffle=False, num_workers=23)


number_of_features = X_train.shape[1]


class LightningModel(L.LightningModule):
    def __init__(self, num_features, activation="ReLU", optimizer_name="Adam", loss_name="MSELoss"):
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
        x = x.view(x.size(0), num_features)
        z = self(x)
        loss = self.get_loss_function()(z, y.unsqueeze(1))
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), num_features)
        z = self(x)
        loss = self.get_loss_function()(z, y.unsqueeze(1))
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return self.get_optimizer()

    def get_optimizer(self):
        optimizers = {
            "Adam": torch.optim.Adam(self.parameters(), lr=0.001),
            "AdamW": torch.optim.AdamW(self.parameters(), lr=0.001),
        }
        return optimizers[self.optimizer_name]

    def get_loss_function(self):
        loss_functions = {
            "MSE": nn.MSELoss(),
            "SmoothL1Loss": nn.SmoothL1Loss(),
            "MAE": nn.L1Loss()
        }
        return loss_functions[self.loss_name]

    def get_activation(self):
        activations = {
            "ReLU": nn.ReLU(),
            "PReLU": nn.PReLU(),
            "ELU": nn.ELU()
        }
        return activations[self.activation]

    def build_model(self):
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


def train_model_tune(config, num_features, train_dataloader_ref, val_dataloader_ref, max_epochs=100):

    model = LightningModel(
        num_features=num_features,
        activation=config["activation"],
        optimizer_name=config["optimizer"],
        loss_name=config["loss_function"]
    )

    metrics = {"loss": "val_loss"}
    callbacks = [TuneReportCheckpointCallback(metrics, on="validation_end")]
    
    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator='auto',
        devices=1,
        callbacks=callbacks,
        enable_progress_bar=False 
    )
    
    trainer.fit(model, train_dataloader_ref, val_dataloader_ref)


def tune_hyperparameters(num_features, train_dataloader_ref, val_dataloader_ref, num_samples=10):
    # Define the search space
    config = {
        "optimizer": tune.choice(["Adam", "AdamW"]),
        "loss_function": tune.choice(["MAE", "MSE","SmoothL1Loss"]),
        "activation": tune.choice(["ReLU", "PReLU", "ELU"]),
    }
    
    search_algo = OptunaSearch(
        metric="loss",
        mode="min"
    )
    
    scheduler = ASHAScheduler(
        time_attr='training_iteration',
        max_t=30,
        grace_period=10,
        reduction_factor=2
    )
    
    tuner = tune.Tuner(
        tune.with_resources(
            partial(
                train_model_tune,
                num_features=num_features,
                train_dataloader_ref=train_dataloader_ref,
                val_dataloader_ref=val_dataloader_ref
            ),
            resources={"cpu": 16, "gpu": 1}  # Adjust based on your hardware
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
    
    # Get best trial
    best_result = results.get_best_result(metric="loss", mode="min")
    best_trial_config = best_result.config
    best_trial_loss = best_result.metrics['loss']
    print(f"Best trial config: {best_trial_config}")
    print(f"Best trial final validation loss: {best_trial_loss}")
    
    return best_trial_config


num_features = X.shape[1]
num_features


best_config = tune_hyperparameters(
        num_features=num_features,
        train_dataloader_ref=train_dataloader,
        val_dataloader_ref=test_dataloader,
        num_samples=18
    )
    