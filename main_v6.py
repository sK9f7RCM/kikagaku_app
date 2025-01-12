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

# Streamlit関連
import streamlit as st
import matplotlib
import matplotlib.pyplot as plt

# ================================================================================================
# ログモジュールによるログの設定
# ================================================================================================
os.makedirs("./log", exist_ok=True) # 既存ログがあってもよい

logging.basicConfig(
    level=logging.INFO, # INFO, WARNING, ERROR, CRITICALを出力(DEBUGは出力しない)
    format='%(asctime)s [%(levelname)s] %(message)s', # ログの形式: 時刻, ログレベル, メッセージ
    handlers=[ # 出力先を制御
        logging.StreamHandler(sys.stdout),              # ログは標準出力に出力する。
        logging.FileHandler("./log/general.log", mode='a', encoding='utf-8')  # ファイル出力 ※追記モード, UTF-8
    ]
)

logger = logging.getLogger(__name__) # 現在のモジュール(__name__)のロガーを取得

# ================================================================================================
# Python標準のWarnings をログに取り込む設定
# ================================================================================================
logging.captureWarnings(True)        # Warningsをログに取り込む
warnings.simplefilter("default")     # 重複するWarningを1度だけ表示。ログには全て出力される

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

# クラスラベルに対応するIDを付与して辞書化
# 例: Writer_Joy_0 → 0, Writer_Joy_1 → 1, ...
dict_label2id = {label_name: idx for idx, label_name in enumerate(ls_labels_32)} # enumerateはリストの要素とIDを返す
dict_id2label = {v: k for k, v in dict_label2id.items()} # 逆引き用辞書 items()はキーと値のペアを返す

# ================================================================================================
# 関数: func_clean_text
# ================================================================================================
def func_clean_text(text: str) -> str:
    # Sentence列に含まれる '_x000D_\n' を通常の改行'\n'に統一し、タブをスペースに変換する。
    text = str(text)
    text = text.replace('_x000D_\n', '\n')
    text = text.replace('\t', ' ')
    return text

# ================================================================================================
# カスタムTrainer: 8感情×4強度(32次元)を一度に学習するために compute_loss をオーバーライド
# ================================================================================================
class class_MultiEmotionTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        (batch_size, 32) の logits と (batch_size, 8) の labels を整合させて
        (batch_size*8, 4) vs. (batch_size*8,) の形で CrossEntropyLoss を計算する。
         余計なキーワード引数(num_items_in_batchなど)は **kwargs で受け取って無視する。
        """
        # 1) inputs から labels を取り出して削除する（model(**inputs) に渡らないようにする）
        labels = inputs.pop("labels")  # shape: (batch_size, 8)

        # 2) Modelのforward呼び出し
        #    ここでは "labels" は渡さず、logitsのみを得る
        outputs = model(**inputs)  # DistilBertForSequenceClassification
        logits = outputs.logits    # shape: (batch_size, 32)

        # 3) カスタムロス計算
        #    (batch_size, 32) → (batch_size*8, 4)
        logits = logits.view(-1, 4)

        #   labels: (batch_size, 8) -> (batch_size*8,)
        labels = labels.view(-1)

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss

# ================================================================================================
# 関数: func_compute_metrics
#   評価指標を計算するため、(batch_size, 32) の logits を 8×4次元に分割して argmax を取る。
# ================================================================================================
def func_compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
    # eval_pred.predictions => (num_samples, 32)
    # eval_pred.label_ids   => (num_samples, 8)
    logits, labels = eval_pred.predictions, eval_pred.label_ids

    # logits がタプルである場合、最初の要素を取り出す
    if isinstance(logits, tuple):
        logits = logits[0]

    num_samples = labels.shape[0]
    # pred_strengths.shape => (num_samples, 8)
    pred_strengths = []
    for i in range(len(ls_emotions)):  # 8感情
        start_idx = i * 4
        end_idx = start_idx + 4
        logits_i = logits[:, start_idx:end_idx]  # (num_samples, 4)
        pred_i = np.argmax(logits_i, axis=1)     # (num_samples,)
        pred_strengths.append(pred_i)
    pred_strengths = np.stack(pred_strengths, axis=1)  # (num_samples, 8)

    # 8感情すべて正解かどうか
    correct_mask = np.all(pred_strengths == labels, axis=1)
    accuracy_all_emotions = correct_mask.mean()

    return {"accuracy_all_emotions": float(accuracy_all_emotions)}

# ================================================================================================
# 関数: func_data_collator
#   デフォルトのDataCollatorWithPaddingを自前実装した例
# ================================================================================================
def func_data_collator(features: list) -> Dict[str, torch.Tensor]:
    input_ids = [f["input_ids"] for f in features]
    attention_mask = [f["attention_mask"] for f in features]
    labels = [f["labels"] for f in features]  # shape=(8,)

    max_len = max(len(x) for x in input_ids)
    padded_input_ids = []
    padded_attention_mask = []
    for ids_, msk_ in zip(input_ids, attention_mask):
        pad_len = max_len - len(ids_)
        padded_input_ids.append(ids_ + [0]*pad_len)
        padded_attention_mask.append(msk_ + [0]*pad_len)

    # labels => (batch_size, 8)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    input_ids_tensor = torch.tensor(padded_input_ids, dtype=torch.long)
    attention_mask_tensor = torch.tensor(padded_attention_mask, dtype=torch.long)

    return {
        "input_ids": input_ids_tensor,
        "attention_mask": attention_mask_tensor,
        "labels": labels_tensor
    }

# ================================================================================================
# 関数: func_tokenize_and_align
#   Sentenceをトークナイズし、8感情の強度を "labels" に格納 (shape: (8,))
# ================================================================================================
def func_tokenize_and_align(dict_batch: Dict[str, Any], tokenizer: AutoTokenizer) -> Dict[str, Any]:
    tokenized = tokenizer(
        dict_batch["Sentence"],
        max_length=512,
        padding="max_length",
        truncation=True
    )
    ls_label_vals = [dict_batch[emo] for emo in ls_emotions]
    tokenized["labels"] = ls_label_vals
    return tokenized

# ================================================================================================
# 関数: func_train_and_eval
#   全体の学習/検証ロジック
# ================================================================================================
def func_train_and_eval(args: argparse.Namespace) -> None:
    logger.info("=== 開始: 学習/検証プロセス ===")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info("GPUを使用します。" if torch.cuda.is_available() else "GPUが使用できないためCPUを使用します。")

    num_train_epochs = args.epoch

    # --------------------------------------------------------------------------------------------
    # データ読み込み (wrime-ver1.tsv)
    # --------------------------------------------------------------------------------------------
    file_path = "../../data/kikagaku_app/wrime-ver1.tsv"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"ファイルが見つかりません: {file_path}")

    df_wrime: pd.DataFrame = pd.read_csv(file_path, sep='\t', encoding='utf-8')
    logger.info(f"元データのサイズ: {df_wrime.shape}")
    logger.info(f"カラム一覧: {df_wrime.columns.tolist()}")

    # サンプル数を制限する（例: 1000サンプルに制限）
    sample_size = 1000  # サンプル数をここで指定
    if len(df_wrime) > sample_size:
        df_wrime = df_wrime.sample(n=sample_size, random_state=42).reset_index(drop=True)
        logger.info(f"データを {sample_size} サンプルに制限しました。")

    # Sentence列のクリーニング
    df_wrime["Sentence"] = df_wrime["Sentence"].apply(func_clean_text)

    # --------------------------------------------------------------------------------------------
    # 学習/検証データに分割(8:2)
    # --------------------------------------------------------------------------------------------
    df_wrime = df_wrime.sample(frac=1.0, random_state=42).reset_index(drop=True)
    split_idx = int(len(df_wrime)*0.8)
    df_train = df_wrime.iloc[:split_idx].reset_index(drop=True)
    df_val = df_wrime.iloc[split_idx:].reset_index(drop=True)

    logger.info(f"学習用データサイズ: {df_train.shape}")
    logger.info(f"検証用データサイズ: {df_val.shape}")

    # --------------------------------------------------------------------------------------------
    # キャッシュ制御
    # --------------------------------------------------------------------------------------------
    cache_dir = "./hf_cache"
    if not args.use_cache:
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
            logger.info("既存キャッシュを削除しました。")

    # --------------------------------------------------------------------------------------------
    # Hugging Face Datasets へ変換するための補助
    # --------------------------------------------------------------------------------------------
    def generate_dataset_from_df(df: pd.DataFrame, tokenizer: AutoTokenizer):
        # dict化
        ls_dict_records = df.to_dict(orient="records")

        # 空のDatasetを作り、mapする
        tmp_dataset = load_dataset(
            "csv",
            data_files={"dummy": file_path},
            split="dummy",
            delimiter="\t",
            cache_dir=cache_dir
        ).from_dict({})  # 一旦空
        # from_dictでメインデータを再構築
        dict_for_dataset = {
            "Sentence": [],
            "Writer_Joy": [],
            "Writer_Sadness": [],
            "Writer_Anticipation": [],
            "Writer_Surprise": [],
            "Writer_Anger": [],
            "Writer_Fear": [],
            "Writer_Disgust": [],
            "Writer_Trust": []
        }
        for row in ls_dict_records:
            dict_for_dataset["Sentence"].append(row["Sentence"])
            for emo in ls_emotions:
                dict_for_dataset[emo].append(int(row[emo]))

        dataset_ = tmp_dataset.from_dict(dict_for_dataset)
        # tokenize
        dataset_ = dataset_.map(
            lambda x: func_tokenize_and_align(x, tokenizer),
            batched=False
        )
        return dataset_

    # --------------------------------------------------------------------------------------------
    # トークナイザ & Dataset生成
    # --------------------------------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained("line-corporation/line-distilbert-base-japanese")
    dataset_train = generate_dataset_from_df(df_train, tokenizer)
    dataset_val = generate_dataset_from_df(df_val, tokenizer)
    dataset_all: DatasetDict = DatasetDict({
        "train": dataset_train,
        "validation": dataset_val
    })

    # --------------------------------------------------------------------------------------------
    # モデルの用意
    #  - num_labels=32
    #  - ただし CrossEntropyLoss の計算はカスタムTrainerでオーバーライド
    # --------------------------------------------------------------------------------------------
    model = AutoModelForSequenceClassification.from_pretrained(
        "line-corporation/line-distilbert-base-japanese",
        num_labels=32,  # 8感情×4強度
        id2label=dict_id2label,
        label2id=dict_label2id
    ).to(device)

    # --------------------------------------------------------------------------------------------
    # TrainingArguments
    # --------------------------------------------------------------------------------------------
    output_dir = "./trainer_output"
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=100,
        per_device_eval_batch_size=100,
        eval_strategy="epoch",  # (deprecated) でも動作はする
        save_strategy="epoch",
        logging_dir="./log",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy_all_emotions",
        greater_is_better=True,
        seed=0,
        fp16=True,                                    # 半精度学習を有効化
        prediction_loss_only=False                    # 評価時にlogitsを保持する
    )

    # --------------------------------------------------------------------------------------------
    # カスタムTrainer(マルチ感情)を実装
    # --------------------------------------------------------------------------------------------
    trainer = class_MultiEmotionTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset_all["train"],
        eval_dataset=dataset_all["validation"],
        tokenizer=tokenizer,
        data_collator=func_data_collator,
        compute_metrics=func_compute_metrics
    )

    # --------------------------------------------------------------------------------------------
    # epoch=0の場合は学習スキップし評価のみ
    # --------------------------------------------------------------------------------------------
    if num_train_epochs == 0:
        logger.info("epoch=0のため、学習スキップし評価のみ実施します。")
        train_metrics = trainer.evaluate(eval_dataset=dataset_all["train"])
        logger.info(f"Train metrics: {train_metrics}")
        val_metrics = trainer.evaluate(eval_dataset=dataset_all["validation"])
        logger.info(f"Val metrics: {val_metrics}")
    else:
        # ----------------------------------------------------------------------------------------
        # 学習実行
        # ----------------------------------------------------------------------------------------
        trainer.train()
        trainer.save_model("./best_model")
        logger.info("学習完了: 最良モデルを ./best_model に保存しました。")

    # --------------------------------------------------------------------------------------------
    # (学習またはepoch=0後) 検証データで最良モデルを使って評価
    # --------------------------------------------------------------------------------------------
    logger.info("最良モデルで検証データを評価します...")
    best_metrics = trainer.evaluate(eval_dataset=dataset_all["validation"])
    logger.info(f"Best model Val metrics: {best_metrics}")

    # --------------------------------------------------------------------------------------------
    # 元データに予測結果を付与してCSV出力
    #   - ./predict_wrime-ver1.csv に保存
    #   - 32クラス確率(softmax後) & 8感情強度
    # --------------------------------------------------------------------------------------------
    logger.info("元データ全件に対して予測し、CSV出力します...")
    model.eval()
    batch_size = 8
    all_logits = []

    for start_idx in range(0, len(df_wrime), batch_size):
        end_idx = start_idx + batch_size
        sub_df = df_wrime.iloc[start_idx:end_idx]
        inputs = tokenizer(
            list(sub_df["Sentence"]),
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_token_type_ids=False  # ★ DistilBertの場合はtoken_type_idsを受け取らないため
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits.cpu().numpy()  # shape=(actual_batch_size, 32)
        all_logits.append(logits)

    all_logits = np.concatenate(all_logits, axis=0)  # (N, 32)

    # softmax -> 8感情ごとの argmax
    pred_strengths = []
    probs_32 = []
    for i in range(len(df_wrime)):
        row_logits = all_logits[i]  # shape=(32,)
        row_probs = []
        row_strengths = []
        for emo_idx in range(len(ls_emotions)):
            chunk = row_logits[4*emo_idx : 4*emo_idx + 4]
            chunk_softmax = np.exp(chunk) / np.sum(np.exp(chunk))
            row_probs.extend(chunk_softmax.tolist())  # 4個の確率
            row_strengths.append(int(np.argmax(chunk_softmax)))
        probs_32.append(row_probs)
        pred_strengths.append(row_strengths)

    # 予測強度を列として追加
    for i, emo in enumerate(ls_emotions):
        df_wrime[f"Predict_{emo}"] = [pred_strengths[row_idx][i] for row_idx in range(len(df_wrime))]

    # 32クラス確率を列として追加
    arr_probs_32 = np.array(probs_32)  # shape=(N, 32)
    for j, label_name in enumerate(ls_labels_32):
        df_wrime[f"Prob_{label_name}"] = arr_probs_32[:, j]

    output_csv = "./predict_wrime-ver1.csv"  # 修正された保存先
    df_wrime.to_csv(output_csv, index=False, encoding="utf-8-sig")
    logger.info(f"予測結果を保存: {output_csv}")

    logger.info("=== 学習/検証プロセスが完了しました ===")


# ================================================================================================
# FastAPIでの推論エンドポイント
# ================================================================================================
class class_InputText(BaseModel):
    text: str

app = FastAPI()

# ローカルに ./best_model があればロードしておく (無ければ警告)
try:
    tokenizer_fastapi = AutoTokenizer.from_pretrained("line-corporation/line-distilbert-base-japanese")
    model_fastapi = AutoModelForSequenceClassification.from_pretrained("./best_model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_fastapi.to(device)
    model_fastapi.eval()
    logger.info("FastAPI用モデルをロードしました。")
except Exception as e:
    logger.warning(f"FastAPI用モデルのロードに失敗: {e}")
    model_fastapi = None
    tokenizer_fastapi = None

@app.post("/predict")
def func_predict(item: class_InputText) -> Dict[str, Any]:
    """
    JSON {"text":"予測したい文章"} を受け取り、32クラス確率と8感情強度を返す。
    """
    if (model_fastapi is None) or (tokenizer_fastapi is None):
        return {"error": "Model is not loaded."}

    text_cleaned = func_clean_text(item.text)
    inputs = tokenizer_fastapi(
        text_cleaned,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model_fastapi(**inputs)
        logits = outputs.logits[0].cpu().numpy()  # shape=(32,)

    dict_probs_32 = {}
    dict_emotion_strength = {}
    for i, emo in enumerate(ls_emotions):
        chunk = logits[4*i : 4*i+4]
        chunk_softmax = np.exp(chunk) / np.sum(np.exp(chunk))

        dict_probs_32[f"{emo}_0"] = float(chunk_softmax[0])
        dict_probs_32[f"{emo}_1"] = float(chunk_softmax[1])
        dict_probs_32[f"{emo}_2"] = float(chunk_softmax[2])
        dict_probs_32[f"{emo}_3"] = float(chunk_softmax[3])

        pred_strength = int(np.argmax(chunk_softmax))
        dict_emotion_strength[emo] = pred_strength

    return {
        "emotion_strengths": dict_emotion_strength,
        "class_probs": dict_probs_32
    }


# ================================================================================================
# Streamlitでの簡易UI
# ================================================================================================
def func_streamlit_app():
    """
    streamlit run main_v6.py --mode app
    などで実行を想定。
    """
    st.title("日本語感情強度予測 (8感情×4強度=32クラス)")
    st.write("文章を入力して「予測実行」を押すと、8感情×4強度の確率を表示します。")

    input_text = st.text_area("入力文章", "")
    if st.button("予測実行"):
        if not input_text.strip():
            st.warning("文章が空です。")
            return

        if (model_fastapi is None) or (tokenizer_fastapi is None):
            st.error("モデルが読み込まれていません。FastAPIが起動していないかもしれません。")
            return

        text_cleaned = func_clean_text(input_text)
        inputs = tokenizer_fastapi(
            text_cleaned,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model_fastapi(**inputs)
            logits = outputs.logits[0].cpu().numpy()  # shape=(32,)

        dict_prob = {}
        dict_strength = {}
        for i, emo in enumerate(ls_emotions):
            chunk = logits[4*i : 4*i+4]
            chunk_softmax = np.exp(chunk) / np.sum(np.exp(chunk))

            dict_prob[f"{emo}_0"] = float(chunk_softmax[0])
            dict_prob[f"{emo}_1"] = float(chunk_softmax[1])
            dict_prob[f"{emo}_2"] = float(chunk_softmax[2])
            dict_prob[f"{emo}_3"] = float(chunk_softmax[3])

            pred_strength = int(np.argmax(chunk_softmax))
            dict_strength[emo] = pred_strength

        # 表示
        st.subheader("予測結果 (8感情の強度)")
        for emo in ls_emotions:
            st.write(f"{emo}: {dict_strength[emo]}")

        # 棒グラフ描画
        st.subheader("32クラスの確率(棒グラフ)")
        matplotlib.rcParams['font.family'] = 'Meiryo'
        fig, ax = plt.subplots(figsize=(8, 6))
        x_labels = list(dict_prob.keys())
        y_vals = list(dict_prob.values())
        ax.bar(x_labels, y_vals)
        plt.xticks(rotation=90, fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)


# ================================================================================================
# main 関数: コマンドライン引数に応じて処理を分岐
# ================================================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train",
                        help="train: 学習実行, app: Streamlitアプリ起動")
    parser.add_argument("--epoch", type=int, default=3,
                        help="学習エポック数(0なら学習スキップで評価のみ)")
    parser.add_argument("--use_cache", type=bool, default=True,
                        help="TrueならHugging Faceのキャッシュ使用, Falseならキャッシュ削除")
    args = parser.parse_args()

    if args.mode == "train":
        func_train_and_eval(args)
    #elif args.mode == "app":
    else:
        func_streamlit_app()
    #else:
    #    logger.info(f"不明なモードです: {args.mode}")
    #    logger.info("train か app を指定してください。")


if __name__ == "__main__":
    main()
