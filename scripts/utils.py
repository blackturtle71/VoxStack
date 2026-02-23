#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data.dataloader import _DatasetKind
from parser import MODELS
import ast
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import lightning as L

def prep_df(path_to_df: str):
    print("[ LOG ] Preparing df for processing...")
    df = pd.read_csv(path_to_df, sep="\t")
    group_col = "uid"
    cols_as_list = MODELS
    df.drop(['utterances', 'timings'], axis=1, inplace=True)

    # read vals as list
    for col in cols_as_list:
        df[col] = df[col].apply(ast.literal_eval)

    agg_dict = {}
    # average all this stuff by utterance
    for col in df.columns:
        if col == group_col:
            continue
        elif col in cols_as_list:
            agg_dict[col] = lambda x: list(np.mean(np.stack(x), axis=0))
        else:
            agg_dict[col] = 'mean'

    df = df.groupby(group_col, as_index=False).agg(agg_dict)
    df.to_csv("data/processed/pitt-chk-1.df", index=False)
    with open("data/processed/pitt-chk-1.info", "w") as fout:
        df.info(buf=fout)
    dem_count = df['dementia'].value_counts().get(1, 0)
    con_count = df['dementia'].value_counts().get(0, 0)
    print(f"[ LOG ] Label distribution:\nDementia: {dem_count}\nControl: {con_count} ")
    return df, dem_count, con_count

class VoxStackDataset(Dataset):
    def __init__(self, df, embedding_column):
        print("[ LOG ] Intitializing VoxStack dataset...")
        self.df = df.reset_index(drop=True)
        self.embedding_column = embedding_column

        # Quantitative markers
        self.quant_cols = [
            "utterance_times",
            "phonological_frags_count",
            "fillers_count"
        ]

        # Embedding columns
        self.embedding_cols = MODELS

        # Acoustic features
        self.acoustic_cols = [
            col for col in df.columns
            if col not in (
                ["uid", "dementia"]
                + self.quant_cols
                + self.embedding_cols
            )
        ]

    def __len__(self):
        return len(self.df)

    def _process_embedding(self, emb):
        if isinstance(emb, str):
            emb = ast.literal_eval(emb)
        return torch.tensor(emb, dtype=torch.float32)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        quant = torch.tensor(
            row[self.quant_cols].values.astype(np.float32)
        )

        acoustic = torch.tensor(
            row[self.acoustic_cols].values.astype(np.float32)
        )

        embedding = self._process_embedding(
            row[self.embedding_column]
        )

        label = torch.tensor(
            row["dementia"],
            dtype=torch.float32
        )

        return {
            "quant": quant,
            "acoustic": acoustic,
            "embedding": embedding,
            "label": label
        }

class VoxStackDataModule(L.LightningDataModule):
    def __init__(self, df, embedding_column, batch_size=32):
        super().__init__()
        self.df = df
        self.embedding_column = embedding_column
        self.batch_size = batch_size

    def setup(self, stage=None):
        train_df, val_df = train_test_split(
            self.df,
            test_size=0.2,
            stratify=self.df["dementia"],
            random_state=42
        )

        quant_cols = [
            "utterance_times",
            "phonological_frags_count",
            "fillers_count"
        ]

        embedding_cols = [
            "BAAI/bge-base-en-v1.5",
            "BAAI/bge-m3",
            "BAAI/bge-large-en-v1.5"
        ]

        acoustic_cols = [
            col for col in self.df.columns
            if col not in (
                ["uid", "dementia"]
                + quant_cols
                + embedding_cols
            )
        ]

        # Scale quant + acoustic together
        scaler = StandardScaler()

        train_df[quant_cols + acoustic_cols] = scaler.fit_transform(
            train_df[quant_cols + acoustic_cols]
        )

        val_df[quant_cols + acoustic_cols] = scaler.transform(
            val_df[quant_cols + acoustic_cols]
        )

        self.train_dataset = VoxStackDataset(
            train_df, self.embedding_column
        )

        self.val_dataset = VoxStackDataset(
            val_df, self.embedding_column
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4
        )

def prep_shap_data(df, emb_col):
    quant_cols = ["utterance_times", "phonological_frags_count", "fillers_count"]
    embedding_cols = MODELS
    acoustic_cols = [c for c in df.columns if c not in ["uid","dementia"] + quant_cols + embedding_cols]

    quant = df[quant_cols].values.astype(np.float32)
    acoustic = df[acoustic_cols].values.astype(np.float32)
    embeddings = []
    for val in df[emb_col].values:
        if isinstance(val, str):
            embeddings.append(ast.literal_eval(val))
        else:
            embeddings.append(val)
    embeddings = np.array(embeddings, dtype=np.float32)
    return quant, acoustic, embeddings


def model_predict(inputs, model):
    device = "cuda"
    model.eval()
    model.to(device)
    quant, acoustic, embedding = inputs

    quant = torch.tensor(quant, dtype=torch.float32, device=device)
    acoustic = torch.tensor(acoustic, dtype=torch.float32, device=device)
    embedding = torch.tensor(embedding, dtype=torch.float32, device=device)
    with torch.no_grad():
        logits = model(quant, acoustic, embedding)
        probs = torch.sigmoid(logits)
    return probs.cpu().numpy()

# courtesy of https://amaarora.github.io/posts/2020-06-29-FocalLoss.html
class WeightedFocalLoss(nn.Module):
    def __init__(self, alpha=.25, gamma=2):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1-alpha]).cuda()
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-BCE_loss)
        F_loss = at*(1-pt)**self.gamma * BCE_loss
        return F_loss.mean()
