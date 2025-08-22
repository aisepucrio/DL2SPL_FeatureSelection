import pandas as pd
import torch
import gc
from utils import LightningModel
import lightning as L
import os

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

    model = LightningModel(num_features=len(top_decision_tree), activation="PReLU", optimizer_name="Adam", loss_name="MSELoss")

    train_size = int(0.8 * len(data))
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]

    del data
    gc.collect()

    train_dataset = LargeDataset(train_data)
    test_dataset = LargeDataset(test_data)

    del train_data, test_data
    gc.collect()

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1024, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1024, shuffle=False, num_workers=4)

    trainer = L.Trainer(
        max_epochs=10,
        accelerator='auto',
        devices=1,
        enable_progress_bar=True
    )

    trainer.fit(model, train_loader, test_loader)

