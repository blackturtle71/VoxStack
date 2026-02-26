#!/usr/bin/env python3
from utils import VoxStackDataset, prep_df, prep_shap_data, model_predict
from parser import MODELS
from torchmetrics.classification import BinaryAUROC, BinaryF1Score
import torch.nn as nn
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import torch
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
from torch.utils.data import DataLoader
import shap
import os
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")

N_FOLDS=10

PATH_TO_DF="data/processed/pitt.tsv"
LOG_PATH="data/logs/"
os.makedirs(LOG_PATH, exist_ok=True)

# clean prev run logs
Path("data/shap.log").unlink(missing_ok=True)
Path("data/train_res.log").unlink(missing_ok=True)

class VoxStack(L.LightningModule):
    def __init__(
        self,
        embedding_dim,
        lr=5e-4,
        weight=None
    ):
        super().__init__()
        self.save_hyperparameters()

        self.quant_net = nn.Sequential(
            nn.Linear(3, 16),
            nn.BatchNorm1d(16),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(16, 8),
            nn.BatchNorm1d(8),
            nn.GELU(),
            nn.Dropout(0.3),
        )

        self.acoustic_net = nn.Sequential(
            nn.Linear(88, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
        )

        self.embedding_net = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.2),
            #nn.Linear(256, 128),
            #nn.GELU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(8 + 64 + 256, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

        if weight is not None:
            self.criterion = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([weight])
            )
        else:
            self.criterion = nn.BCEWithLogitsLoss()

        self.auroc = BinaryAUROC()
        self.f1 = BinaryF1Score()

    def forward(self, quant, acoustic, embedding):
        q = self.quant_net(quant)
        a = self.acoustic_net(acoustic)
        e = self.embedding_net(embedding)

        fused = torch.cat([q, a, e], dim=1)
        logits = self.classifier(fused)
        return logits

    def training_step(self, batch, batch_idx):
        logits = self(
            batch["quant"],
            batch["acoustic"],
            batch["embedding"]
        )
        y = batch["label"]

        loss = self.criterion(logits.squeeze(1), y)

        probs = torch.sigmoid(logits).squeeze(1)

        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        self.auroc.update(probs, y.int())
        self.log("train_auc", self.auroc, prog_bar=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(
            batch["quant"],
            batch["acoustic"],
            batch["embedding"]
        )
        y = batch["label"]

        loss = self.criterion(logits.squeeze(1), y)
        probs = torch.sigmoid(logits).squeeze(1)

        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.auroc.update(probs, y.int())
        self.log("val_auc", self.auroc(probs, y.int()), prog_bar=True, on_epoch=True)
        self.f1.update(probs, y.int())
        self.log("val_f1", self.f1(probs, y.int()), prog_bar=True, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=2
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_auc"
            }
        }

def evaluate_model(model, dataloader):
    model.eval().float()
    preds, targets = [], []

    with torch.no_grad():
        for batch in dataloader:
            logits = model(batch["quant"], batch["acoustic"], batch["embedding"])
            probs = torch.sigmoid(logits)
            preds.extend(probs.cpu().numpy())
            targets.extend(batch["label"].cpu().numpy())

    auc = roc_auc_score(targets, preds)
    f1 = f1_score(targets, (np.array(preds) > 0.5).astype(int))
    return auc, f1

def run_trainer():
    df = prep_df(PATH_TO_DF)

    # takes into account that the same uid shouldn't be in val_df and train_df at the same time
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True)
    labels = df["dementia"].values

    val_dfs_per_fold = {emb_model: [] for emb_model in MODELS}

    for emb_model_name in MODELS:
        print(f"\n[LOG] Training with embedding: {emb_model_name}")
        aucs, f1s = [], []

        sample_emb = df[emb_model_name].iloc[0]
        if isinstance(sample_emb, str):
            import ast
            sample_emb = ast.literal_eval(sample_emb)

        emb_dim = len(sample_emb)
        print(f"[LOG] Detected embedding dimension: {emb_model_name}")

        fold_idx = 0
        for train_idx, val_idx in sgkf.split(df, y=labels, groups=df["uid"]):
            safe_name = emb_model_name.split("/")[1] if "/" in emb_model_name else emb_model_name
            l_logger = TensorBoardLogger(
                save_dir=LOG_PATH,
                name=f"{safe_name}_fold{fold_idx}"
            )
            fold_idx += 1
            print(f"[LOG] Fold {fold_idx}")

            train_df = df.iloc[train_idx].reset_index(drop=True)
            val_df = df.iloc[val_idx].reset_index(drop=True)
            val_dfs_per_fold[emb_model_name].append(val_df.copy())

            train_patients = set(train_df["uid"])
            val_patients = set(val_df["uid"])

            assert len(train_patients.intersection(val_patients)) == 0, "DATA LEAK!!! UID exists both in val_df and train_df!"

            # calculate weights (per fold calc is safer)
            dem_count = train_df["dementia"].sum()
            con_count = len(train_df) - dem_count
            weight = min(dem_count, con_count) / max(dem_count, con_count)

            # Scale quantitative + acoustic features
            quant_cols = ["utterance_times", "phonological_frags_count", "fillers_count"]
            embedding_cols = MODELS
            acoustic_cols = [c for c in df.columns if c not in ["uid","dementia"] + quant_cols + embedding_cols]

            scaler = StandardScaler()
            train_df[quant_cols + acoustic_cols] = scaler.fit_transform(train_df[quant_cols + acoustic_cols])
            val_df[quant_cols + acoustic_cols] = scaler.transform(val_df[quant_cols + acoustic_cols])

            train_dataset = VoxStackDataset(train_df, emb_model_name)
            val_dataset = VoxStackDataset(val_df, emb_model_name)

            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

            # Initialize model
            model = VoxStack(
                embedding_dim=emb_dim,
                weight=weight
            )

            # Trainer
            trainer = L.Trainer(
                max_epochs=50,
                accelerator="gpu",
                devices=1,
                logger=l_logger,
                callbacks=[
                    EarlyStopping(
                        monitor="val_auc",
                        patience=5,
                        mode="max"
                    ),
                    ModelCheckpoint(
                        dirpath="data/checkpoints/",
                        filename=f"voxstack_{safe_name}_fold{fold_idx}",
                        save_top_k=1,
                        monitor="val_auc",
                        mode="max",
                        save_weights_only=True
                    )
                ],
                enable_checkpointing=True
            )

            # Train
            trainer.fit(model, train_loader, val_loader)

            # Evaluate
            auc, f1 = evaluate_model(model, val_loader)
            print(f"[LOG] Fold {fold_idx} AUC={auc:.3f}, F1={f1:.3f}")
            aucs.append(auc)
            f1s.append(f1)

            # Average metrics across folds
            with open ("data/train_res.log", "a") as fout:
                print(f"[RESULT] Embedding: {emb_model_name} | Avg AUC={np.mean(aucs):.3f}, Avg F1={np.mean(f1s):.3f}")
                fout.write(f"[RESULT] Embedding: {emb_model_name} | Avg AUC={np.mean(aucs):.3f}, Avg F1={np.mean(f1s):.3f}\n")

    return val_dfs_per_fold

def run_shap(val_dfs_per_fold: dict):
    device = "cuda"
    bg_size = 50
    topk_emb = 10

    def _aggregate_branch(shap_list):
        fold_importances = []
        for fold_vals in shap_list:
            fold_importance = np.mean(np.abs(fold_vals), axis=0)
            fold_importances.append(fold_importance)
        return np.mean(fold_importances, axis=0)

    for emb_model_name in MODELS:
        print(f"\n[LOG] Aggregating SHAP for embedding: {emb_model_name}")
        shap_quant_all, shap_acoustic_all, shap_embedding_all = [], [], []

        for fold_idx in range(1, N_FOLDS+1):
            val_df = val_dfs_per_fold[emb_model_name][fold_idx-1]

            safe_name = emb_model_name.split("/")[1] if "/" in emb_model_name else emb_model_name
            model_path = f"data/checkpoints/voxstack_{safe_name}_fold{fold_idx}.ckpt"
            model = VoxStack.load_from_checkpoint(model_path,
                                                  embedding_dim=len(prep_shap_data(val_df,emb_model_name)[2][0]),
                                                  weights_only=False)
            model.to(device).float()
            model.eval()

            quant_val, acoustic_val, emb_val = prep_shap_data(val_df, emb_model_name)

            bg_idx = np.random.choice(len(val_df), min(bg_size, len(val_df)), replace=False)
            quant_bg = torch.tensor(quant_val[bg_idx]).to(device)
            acoustic_bg = torch.tensor(acoustic_val[bg_idx]).to(device)
            emb_bg = torch.tensor(emb_val[bg_idx]).to(device)

            explainer = shap.DeepExplainer(model, [quant_bg, acoustic_bg, emb_bg])
            quant_tensor = torch.tensor(quant_val).to(device)
            acoustic_tensor = torch.tensor(acoustic_val).to(device)
            emb_tensor = torch.tensor(emb_val).to(device)

            shap_vals = explainer.shap_values(
                [quant_tensor, acoustic_tensor, emb_tensor]
            )

            shap_quant_all.append(shap_vals[0])
            shap_acoustic_all.append(shap_vals[1])
            shap_embedding_all.append(shap_vals[2])

        shap_quant_mean = _aggregate_branch(shap_quant_all)
        shap_acoustic_mean = _aggregate_branch(shap_acoustic_all)
        shap_embedding_mean = _aggregate_branch(shap_embedding_all)

        quant_cols = ["utterance_times", "phonological_frags_count", "fillers_count"]
        embedding_cols = MODELS
        acoustic_cols = [c for c in val_df.columns if c not in ["uid","dementia"] + quant_cols + embedding_cols]

        with open("data/shap.log", "a") as fout:
            fout.write(f"\n[LOG] Aggregating SHAP for embedding: {emb_model_name}\n")
            print("\nQuantitative Feature Importances:")
            shap_quant_mean = np.array(shap_quant_mean).flatten()
            for name, val in zip(quant_cols, shap_quant_mean):
                print(f"{name}: {val:.5f}")
                fout.write(f"{name}: {val:.5f}\n")

            print("\nAcoustic Feature Importances (Top 10):")
            shap_acoustic_mean = np.array(shap_acoustic_mean).flatten()
            top10_idx = np.argsort(shap_acoustic_mean)[-10:][::-1]
            for idx in top10_idx:
                print(f"{acoustic_cols[idx]}: {shap_acoustic_mean[idx]:.5f}")
                fout.write(f"{acoustic_cols[idx]}: {shap_acoustic_mean[idx]:.5f}\n")

            print(f"\nTop-{topk_emb} Embedding Dimensions:")
            shap_embedding_mean = np.array(shap_embedding_mean).flatten()
            topk_idx = np.argsort(shap_embedding_mean)[-topk_emb:][::-1]
            for i in topk_idx:
                print(f"emb_{i}: {shap_embedding_mean[i]:.5f}")
                fout.write(f"emb_{i}: {shap_embedding_mean[i]:.5f}\n")

            quant_score = np.mean(shap_quant_mean)
            acoustic_score = np.mean(shap_acoustic_mean)
            embedding_score = np.mean(shap_embedding_mean)

            branch_scores = {
                "Quantitative": quant_score,
                "Acoustic": acoustic_score,
                "Embedding": embedding_score
            }

            branch_scores = dict(sorted(branch_scores.items(), key=lambda x: x[1], reverse=True))
            print("Branch-level mean SHAP importance:")
            for branch, score in branch_scores.items():
                print(f"{branch}: {score:.5f}")
                fout.write(f"{branch}: {score:.5f}\n")

if __name__ == "__main__":
    val_dfs_per_fold = run_trainer()
    run_shap(val_dfs_per_fold)
