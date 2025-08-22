import pandas as pd
import torch
import gc
from utils import LightningModel
import lightning as L
import os
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback

class LargeDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data.iloc[idx, :-1].values, dtype=torch.float32)
        y = torch.tensor(self.data.iloc[idx, -1], dtype=torch.float32)
        return x, y

def select_top_features(df, top_percentage=0.7):
    df_sorted = df.sort_values(by='importance', ascending=False)
    top_n = int(len(df_sorted) * top_percentage)
    return df_sorted.head(top_n)['feature'].tolist()

def train_model_tune(config, train_loader, test_loader, num_features):
    model = LightningModel(
        num_features=num_features,
        activation=config['activation'],
        optimizer_name=config['optimizer'],
        loss_name=config['loss']
    )

    trainer = L.Trainer(
        max_epochs=10,
        accelerator='auto',
        devices=1,
        enable_progress_bar=False,
        callbacks=[TuneReportCheckpointCallback({'val_loss': 'val_loss'}, on='validation_end')]
    )

    trainer.fit(model, train_loader, test_loader)

if __name__ == '__main__':
    decision_tree = pd.read_csv('feature_importances_DecisionTree.csv')
    random_forest = pd.read_csv('feature_importances_RandomForest.csv')
    gradient_boosting = pd.read_csv('feature_importances_GradientBoosting.csv')
    xgboost = pd.read_csv('feature_importances_XGBoost.csv')

    top_percentage = 0.7
    top_decision_tree = select_top_features(decision_tree, top_percentage)
    top_random_forest = select_top_features(random_forest, top_percentage)
    top_gradient_boosting = select_top_features(gradient_boosting, top_percentage)
    top_xgboost = select_top_features(xgboost, top_percentage)
    
    target_column = 'perf'
    columns_to_read = top_decision_tree + [target_column]
    data = pd.read_parquet('data.parquet', columns=columns_to_read)
    data = data.astype('float32')
    gc.collect()

    train_size = int(0.8 * len(data))
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]

    del data
    gc.collect()

    train_dataset = LargeDataset(train_data)
    test_dataset = LargeDataset(test_data)

    del train_data, test_data
    gc.collect()

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=2)

    search_space = {
        'activation': tune.grid_search(['ReLU', 'PReLU', 'ELU']),
        'optimizer': tune.grid_search(['Adam', 'AdamW']),
        'loss': tune.grid_search(['MSELoss', 'L1Loss', 'SmoothL1Loss'])
    }

    scheduler = ASHAScheduler(
        max_t=10,
        grace_period=1,
        reduction_factor=2
    )

    tuner = tune.Tuner(
        tune.with_parameters(
            train_model_tune,
            train_loader=train_loader,
            test_loader=test_loader,
            num_features=len(top_decision_tree)
        ),
        param_space=search_space,
        tune_config=tune.TuneConfig(
            metric='val_loss',
            mode='min',
            scheduler=scheduler
        )
    )

    results = tuner.fit()

    best_result = results.get_best_result(metric='val_loss', mode='min')
    print("Best Config:", best_result.config)
    print("Best Validation Loss:", best_result.metrics['val_loss'])