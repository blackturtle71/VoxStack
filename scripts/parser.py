#!/usr/bin/env python3

import pandas as pd
import re
from pathlib import Path
from FlagEmbedding import FlagAutoModel
from tqdm import tqdm
from pydub import AudioSegment
import opensmile
import os
import numpy as np
from transformers import AutoModel, AutoTokenizer
import torch

DATA_PATHS = [("data/unprocessed/DementiaBank/Pitt/Dementia/cookie/","data/unprocessed/DementiaBank/Pitt_audio/Dementia/cookie/"),
              ("data/unprocessed/DementiaBank/Pitt/Control/cookie/","data/unprocessed/DementiaBank/Pitt_audio/Control/cookie/")]

MODELS = ['BAAI/bge-base-en-v1.5', 'BAAI/bge-m3', 'BAAI/bge-large-en-v1.5',
          'bert-base-uncased', 'bert-large-uncased', 'roberta-base',
          'roberta-large']

def parse_cha(dir_path: str):
    second_last_dir = os.path.basename(os.path.dirname(os.path.dirname(dir_path)))
    if "Dementia" in second_last_dir:
        print("[ LOG ] Processing 'Dementia' transcript")
        flag = 1
    elif "Control" in second_last_dir:
        print("[ LOG ] Processing 'Control' transcript")
        flag = 0
    else:
        print("[ WARNING ] Unknown transcript category")
        return None

    path = Path(dir_path)
    files = [f for f in path.iterdir() if f.is_file()]
    uid = []
    utterances = []
    utterance_times = []
    timings = []
    phonological_frags_count = []
    fillers_count = []
    speech_event_count = [] # like growls, laughs and so on

    for file in files:
        with open(file, 'r', encoding='utf-8', errors="ignore") as fin:
            for line in fin:
                if line.startswith("*PAR"):
                    # set uid
                    uid.append(file.name[:-4])
                    # capture timings
                    match = re.search(r'\x15(\d+)_(\d+)\x15', line)
                    timing = (int(match.group(1)), int(match.group(2))) if match else None
                    utterance_times.append(timing[1] - timing[0] if timing else None)
                    timings.append(timing)

                    line = re.sub(r'[\t\n\r]', ' ', line) # drop formatting characters
                    line = re.sub(r'^\*PAR:', '', line) # drop PAR marker
                    line = re.sub(r'\[[^\]]*\]', '', line) # drop anything inside []
                    line = re.sub(r'\x15\d+_\d+\x15', '', line) # drop timings
                    line = re.sub(r'www', '', line) # drop www
                    line = re.sub(r'\b[x]+\b', '', line) # drop x xx xxx and so on

                    # capture phonological frags count
                    matches = re.findall(r'&\+\S+', line)
                    phonological_frags_count.append(len(matches))
                    line = re.sub(r'&\+\S+', '', line)

                    # capture fillers count
                    matches = re.findall(r'&\-\S+', line)
                    fillers_count.append(len(matches))
                    line = re.sub(r'&\-', '', line)

                    # capture speech event count
                    matches = re.findall(r'&\=\S+', line)
                    speech_event_count.append(len(matches))
                    line = re.sub(r'&\=\S+', '', line)

                    line = re.sub(r'\([^\)]*\)', '', line) # drop incomplete parts of the word, i.e. goin(g) to goin

                    # drop prosody
                    line = line.replace(":", "")
                    line = line.replace("ˈ", "")
                    line = line.replace("ˌ", "")
                    line = line.replace("^", "")
                    line = line.replace("≠", "")

                    # drop tone
                    line = line.replace("↑", "")
                    line = line.replace("↓", "")

                    # drop satellite markers
                    line = line.replace("‡", "")
                    line = line.replace("„", "")

                    # drop scope for scoped symbols
                    line = line.replace("<", "")
                    line = line.replace(">", "")

                    # drop leftover symbols
                    line = line.replace("@l", " ") # letter pronounciation mark
                    line = line.replace("@", " ") # leftover @
                    line = line.replace("/", " ")
                    line = line.replace("_", " ")
                    line = line.replace("+", " ")

                    line = re.sub(r'\s+', ' ', line).strip() # colapse multiple spaces (from cleaning)
                    utterances.append(line)


    parsed = {"uid": uid,
              "dementia": flag,
              "utterances": utterances,
              "utterance_times": utterance_times,
              "timings": timings,
              "phonological_frags_count": phonological_frags_count,
              "fillers_count": fillers_count
              }

    df = pd.DataFrame(parsed)
    df = df.sort_values(by="uid")
    missing_timings = df["timings"].isna().sum()
    if missing_timings != 0:
        print(f"[ WARNING ] {missing_timings} utterances have no timings. Removing...")
        df.dropna(subset=["timings"], inplace=True)

    return df

#def _load_model(model_name: str):
#    model = FlagAutoModel.from_finetuned(
#        model_name
#    )
#    return model

#def embed(utterance: str, model: FlagAutoModel):
#    emb = model.encode(utterance)
#    if isinstance(emb, dict):
#        return emb['dense_vecs'].tolist()
#    elif isinstance(emb, np.ndarray):
#        return emb.tolist()

def _load_model(model_name: str, device: str ="cuda"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return tokenizer, model

def embed(utterance: str, tokenizer, model, device: str="cuda"):
    inputs = tokenizer(utterance, return_tensors='pt', truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    emb = outputs.last_hidden_state[:, 0, :]
    emb = emb.squeeze(0).cpu().numpy()

    return emb.tolist()

def _chunk_audio(df: pd.DataFrame, dir_path: str):
    second_last_dir = os.path.basename(os.path.dirname(os.path.dirname(dir_path)))
    if "Dementia" in second_last_dir:
        print("[ LOG ] Processing 'Dementia' audio")
        save_path = "data/processed/chunked_audio/dem/"
    elif "Control" in second_last_dir:
        print("[ LOG ] Processing 'Control' audio")
        save_path = "data/processed/chunked_audio/con/"
    else:
        print("[ WARNING ] Unknown audio category")
        save_path = "data/processed/chunked_audio/unknown/"

    os.makedirs(save_path, exist_ok=True)
    lost_files = []
    for uid in tqdm(df['uid'].unique(), desc="Chunking audio"):
        try:
            audio = AudioSegment.from_mp3(f"{dir_path}{uid}.mp3")
            chunks = []
            combined = AudioSegment.empty()
            uid_slice = df.loc[df['uid'] == uid]

            for row in uid_slice.itertuples(index=False):
                timings = row.timings
                start = timings[0]
                end = timings[1]
                chunks.append(audio[start:end])

            for chunk in chunks:
                combined += chunk

            combined.export(f"{save_path}{uid}.mp3", format="mp3")
        except FileNotFoundError:
            lost_files.append(uid)

    if lost_files != []:
        print(f"[ WARNING ] You have {len(lost_files)} missing files: {lost_files}")

    return save_path

def extract_audio_feats(df: pd.DataFrame, dir_path: str):
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    chunked_audio_dir = _chunk_audio(df, dir_path)
    path = Path(chunked_audio_dir)
    files = [f for f in path.iterdir() if f.is_file()]
    full_set = []
    for file in tqdm(files, desc="Extracting audio features"):
        features = smile.process_file(file)
        features['uid'] = file.name[:-4]
        full_set.append(features)
    full_set = pd.concat(full_set, ignore_index=True)
    df = pd.merge(df, full_set, on="uid")
    return df

        
def run():
    # preload models
    for model_name in MODELS:
        print(f"[ LOG ] Preloading model: {model_name}")
        _load_model(model_name)

    df_list = []
    for dir_path in DATA_PATHS:
        print(f"[ LOG ] Parsing {dir_path[0]}")
        df = parse_cha(dir_path[0])

        if df is None:
            return

        df = extract_audio_feats(df, dir_path[1])

        for model_name in MODELS:
            print(f"[ LOG ] Running embedding process for {model_name}")
            tokenizer, model = _load_model(model_name)
            embs = []
            for utterance in tqdm(df['utterances'], desc="Embedding utterances"):
                emb = embed(utterance, tokenizer, model)
                embs.append(emb)
            df[model_name] = embs

        df_list.append(df)

    df = pd.concat(df_list)
    df.to_csv("data/processed/pitt.tsv", sep='\t', index=False)

if __name__ == "__main__":
    run()
