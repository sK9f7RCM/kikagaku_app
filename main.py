import logging # ロギングモジュールのインポート

import os
import sys
import argparse
import warnings
import shutil
from pathlib import Path
from typing import Dict, Any, Tuple

import torch
from torch import nn
from torch.utils.data import Dataset

import pandas as pd
import numpy as np

# transformers関連
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from transformers.trainer_utils import EvalPrediction

# Hugging Face Datasets
from datasets import load_dataset, DatasetDict

# FastAPI関連
from fastapi import FastAPI
from pydantic import BaseModel

# ================================================================================================
# グローバル設定
# ================================================================================================
ls_emotions = [ # 8感情
    'Writer_Joy', 'Writer_Sadness', 'Writer_Anticipation', 'Writer_Surprise',
    'Writer_Anger', 'Writer_Fear', 'Writer_Disgust', 'Writer_Trust'
]

# 分類予測対象の32クラス (8感情*4強度) のラベルを作成
ls_labels_32 = []
for emo in ls_emotions:
    for i in range(4): # 強度0,1,2,3
        ls_labels_32.append(f"{emo}_{i}") # 例: Writer_Joy_0, Writer_Joy_1, ...

# クラスに対応するIDを付与して辞書化
# 例: Writer_Joy_0 → 0, Writer_Joy_1 → 1, ...
dict_label2id = {label_name: idx for idx, label_name in enumerate(ls_labels_32)} # enumerateはリストの要素とIDを返す
dict_id2label = {v: k for k, v in dict_label2id.items()} # 逆引き用辞書 items()はキーと値のペアを返す

# ================================================================================================
# Sentenceの前処理
# ================================================================================================
def func_clean_text(text: str) -> str:
    # Sentence列に含まれる '_x000D_\n' を通常の改行'\n'に統一し、タブをスペースに変換する。
    text = str(text)
    text = text.replace('_x000D_\n', '\n')
    return text

# ================================================================================================
# FastAPIエンドポイントの定義
# ================================================================================================
class InputText(BaseModel): # リクエストデータの構造を定義し、自動的に検証
    text: str # 入力データが文字列であることを指定

app = FastAPI()

@app.post("/predict") # POSTリクエストを受け取るエンドポイント

# ================================================================================================
#  - 入力テキストを受け取り、予測結果を返す関数を定義
# ================================================================================================
def func_fastapi_predict(text: str) -> Dict[str, Any]:

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # デバイスの設定

    # 最良モデルをロードする。
    best_model_path = "./best_model"
    if os.path.exists(best_model_path):
        model = AutoModelForSequenceClassification.from_pretrained("./best_model").to(device)
    else:
        raise FileNotFoundError(f"保存されたモデルが見つかりません: {best_model_path}") # 保存されたモデルがない場合はエラーを出力して終了
    model.eval()

    text_cleaned = func_clean_text(text) # 入力テキストのクリーニング
    tokenizer = AutoTokenizer.from_pretrained("line-corporation/line-distilbert-base-japanese", trust_remote_code=True)
    inputs = tokenizer( # トークン化
        text_cleaned,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).to(device) # GPU or CPUに転送

    with torch.no_grad(): # 勾配計算を行わないようにする。(予測、評価時に適切)
        outputs = model(**inputs) # モデルに入力を渡して予測
        logits = outputs.logits[0].cpu().numpy()  # GPU上の予測結果テンソルをCPUへ移動し、numpy形式に変換 shape=(32,)

    dict_probs_32 = {}
    dict_emotion_strength = {}
    for i, emo in enumerate(ls_emotions): # 1感情毎に処理
        chunk = logits[4*i : 4*i+4] # 1感情の4強度
        chunk_softmax = np.exp(chunk) / np.sum(np.exp(chunk)) # 4強度の予測値を合計1になるように、変換

        # 4強度の確率を辞書に追加
        dict_probs_32[f"{emo}_0"] = float(chunk_softmax[0])
        dict_probs_32[f"{emo}_1"] = float(chunk_softmax[1])
        dict_probs_32[f"{emo}_2"] = float(chunk_softmax[2])
        dict_probs_32[f"{emo}_3"] = float(chunk_softmax[3])

        pred_strength = int(np.argmax(chunk_softmax)) # 最大確率の強度
        dict_emotion_strength[emo] = pred_strength # 最大確率の強度を辞書に追加

    return {
        "emotion_strengths": dict_emotion_strength, # 8感情の強度
        "class_probs": dict_probs_32 # 32クラスの確率
    }