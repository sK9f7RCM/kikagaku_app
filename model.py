# Version: 6

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
import requests

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
# カスタムTrainer: 8感情それぞれについて、4強度のクラス分類を行うために、logitsとlabelsの形式を変更する必要がある。 
#   カスタムの損失関数をオーバーライド
# ================================================================================================
# Hugging Face の transformers ライブラリにおけるモデルの学習と評価を行うためのクラス Trainer を継承
class class_MultiEmotionTrainer(Trainer): 
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs): # 損失計算をカスタマイズ
        # 通常の Trainer では、モデルの予測出力logitsが (batch_size, クラス数) = (batch_size, 32) の形式になる。
        # 正解ラベルlabelsは、8感情それぞれの感情強度がラベルとして格納されており、(batch_size, クラス数) = (batch_size, 8) の形式になっている。
        # 8感情それぞれについて、4強度のクラス分類を行うために、形式を変更する必要がある。

        # 余計なキーワード引数(num_items_in_batchなど)は **kwargs で受け取って無視する。

        # 1) inputs から "labels" キーを取り出して削除する（modelに渡らないようにする）
        # 入力辞書inputsに "labels" キーがあると、DistilBertForSequenceClassification 側が自動でロスを計算しようとする仕様があるため、
        #「(batch_size, 8) のラベル」を見て、「単一タスクの (batch_size,) ラベル」を想定した内部ロジックと衝突し、形状の不一致が起きます。
        labels = inputs.pop("labels")  # shape: (batch_size, 8)

        # 2) Modelのforward呼び出し
        #    ここでは "labels" は渡さず、logitsのみを得る
        outputs = model(**inputs)  # DistilBertForSequenceClassificationモデルを使用。**inputsで辞書のキー, 値を展開してmodel()に渡す。
        logits = outputs.logits    # shape: (batch_size, 32)

        # 3) カスタムロス計算
        # (batch_size, 32) → (batch_size*8, 4)
        # logits[0] = [Joy_0, Joy_1, Joy_2, Joy_3, Sadness_0, Sadness_1, ..., Trust_3]
        # →
        # logits[0] -> [Joy_0, Joy_1, Joy_2, Joy_3]
        # logits[1] -> [Sadness_0, Sadness_1, Sadness_2, Sadness_3]
        # ...
        logits = logits.view(-1, 4) # -1 は「バッチサイズに基づき自動計算される次元」を意味する。

        # labels: (batch_size, 8) -> (batch_size*8,)
        # labels[0] = [Joy, Sadness, ..., Trust]
        # →
        # labels[0] -> [Joy] Joy_0, Joy_1, Joy_2, Joy_3 のうちの正解ラベル
        # labels[1] -> [Sadness] Sadness_0, Sadness_1, Sadness_2, Sadness_3のうちの正解ラベル
        # ...
        labels = labels.view(-1) # -1 は「バッチサイズに基づき自動計算される次元」を意味する。

        loss_fct = nn.CrossEntropyLoss() # 4強度についての多クラス分類なので、交差エントロピー損失関数を使用。小さいほど正確な予測。
        loss = loss_fct(logits, labels) # softmax関数により4強度の予測値合計を1に正規化した後、正解ラベル(正解の感情強度)の分類確率を取得。負の対数尤度 loss=-log(正解の感情強度の分類確率) 。
        # loss: (batch_size*8,)
        # loss[0] -> [Joy] 損失
        # loss[1] -> [Sadness] 損失
        # ...
        # バッチ全体の平均損失 = (データ点1の損失 + データ点2の損失 + ...) / データ点数

        return (loss, outputs) if return_outputs else loss # デフォルトでは損失のみを返す。return_outputs=Trueの場合は損失とモデル出力を返す。

# ================================================================================================
#   カスタムの評価関数を定義。Trainerに取り込む外部関数として指定できるようになっている。
# ================================================================================================
def func_compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]: # EvalPrediction は Trainer が評価中に渡すオブジェクト。
    logits = eval_pred.predictions # (batch_size, 32)
    labels = eval_pred.label_ids # (batch_size, 8)

    # モデルの予測値以外の情報が含まれていることで logits がタプルである場合、実際の予測値は logits[0] に格納されているので、取り出す
    if isinstance(logits, tuple):
        logits = logits[0]

    # 損失計算のときと同様に、logitsとlabelsの形式を変更。
    # logits: (batch_size, 32) -> (batch_size*8, 4)
    logits = logits.reshape(-1, 4)

    # labels: (batch_size, 8) -> (batch_size*8,)
    labels = labels.reshape(-1)

    # 正解率を計算。
    predictions = torch.argmax(torch.tensor(logits), dim=-1) # 最後の次元(4強度)に対してargmaxを取る
    correct = (predictions == torch.tensor(labels)).sum().item() # 損失計算のときとは異なり、分類確率の大きさは考慮されない。最大確率をとるラベルが正解と一致するかだけ。
    accuracy = correct / len(labels) # 正解数 / データ点数

    return {"accuracy_all_emotions": float(accuracy)} # 正解率を返す

# ================================================================================================
#   カスタムのデータコレーター関数を定義。Trainerに取り込む外部関数として指定できるようになっている。
#   動的paddingによりメモリ節約。
#   labelsをtorch形式のテンソルで返すように拡張。
# ================================================================================================
def func_data_collator(features: list) -> Dict[str, torch.Tensor]:
    # features: List[Dict[str, 情報]] データ点情報がまとまった辞書のリスト(1辞書が1Sentenceに対応)
    input_ids = [f["input_ids"] for f in features] # トークン化されたSentence (全Sentenceなのでテンソル)
    attention_mask = [f["attention_mask"] for f in features] # paddingされた部分を無視するためのマスク(全Sentenceなのでテンソル)
    labels = [f["labels"] for f in features]  # 8感情の正解の感情強度ラベル shape=(8,) (全Sentenceなのでテンソル)

    max_len = max(len(x) for x in input_ids) # 全Sentenceのうち、最大のトークン数を取得
    padded_input_ids = []
    padded_attention_mask = []
    for ids_, msk_ in zip(input_ids, attention_mask): # 最大トークン数に合わせて各Sentence毎にpadding
        pad_len = max_len - len(ids_) # paddingする長さ
        padded_input_ids.append(ids_ + [0]*pad_len) # トークンを0でpadding
        padded_attention_mask.append(msk_ + [0]*pad_len) # マスクを0でpadding

    # labels => (batch_size, 8)
    labels_tensor = torch.tensor(labels, dtype=torch.long) # torch形式のテンソルに変換
    input_ids_tensor = torch.tensor(padded_input_ids, dtype=torch.long) # torch形式のテンソルに変換
    attention_mask_tensor = torch.tensor(padded_attention_mask, dtype=torch.long) # torch形式のテンソルに変換

    return {"input_ids": input_ids_tensor, "attention_mask": attention_mask_tensor, "labels": labels_tensor}

# ================================================================================================
#   batch内辞書のSentenceをトークン化し、8感情の強度を "labels" に格納 shape=(8,)
# ================================================================================================
def func_tokenize_and_align(dict_batch: Dict[str, Any], tokenizer: AutoTokenizer) -> Dict[str, Any]:
    tokenized = tokenizer(
        dict_batch["Sentence"],
        max_length=512, #BERT系モデルの最大トークン数
        padding=False, #paddingはfunc_data_collatorで行う
        truncation=True #max_lengthを超えた場合にトークンを末尾からトリミング
    )
    ls_label_vals = [dict_batch[emo] for emo in ls_emotions] # batch内の正解ラベルを取得
    tokenized["labels"] = ls_label_vals # トークン化後のbatch内辞書に正解ラベルを追加
    return tokenized

# ================================================================================================
#   学習,検証の定義
# ================================================================================================
def func_train_and_eval(args: argparse.Namespace) -> None: # コマンドライン引数を受け取る
    logger.info("=== 開始: 学習,検証プロセス ===") #　メッセージをログへ出力

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # デバイスの設定
    logger.info("GPUを使用します。" if torch.cuda.is_available() else "GPUが使用できないためCPUを使用します。") #　メッセージをログへ出力

    num_train_epochs = args.epoch # Epoch数をコマンドライン引数から取得

    # --------------------------------------------------------------------------------------------
    # データ読み込み (wrime-ver1.tsv)
    # --------------------------------------------------------------------------------------------
    file_path = "../../data/kikagaku_app/wrime-ver1.tsv" # 元データファイルパス
    if not os.path.exists(file_path): # ファイルが存在しない場合はエラーを出力
        raise FileNotFoundError(f"ファイルが見つかりません: {file_path}")

    df_wrime = pd.read_csv(file_path, sep='\t', encoding='utf-8') # .tsvファイルをDataFrameとして読み込む
    logger.info(f"元データのサイズ: {df_wrime.shape}") #　メッセージをログへ出力
    logger.info(f"カラム一覧: {df_wrime.columns.tolist()}") #　メッセージをログへ出力

    # サンプル数を制限する
    #frac = 0.05 # 元データに対するサンプル割合
    frac = 0.5 # 元データに対するサンプル割合
    sample_size = int(len(df_wrime) * frac) # サンプル数
    df_wrime = df_wrime.sample(frac=frac, random_state=42).reset_index(drop=True) # ランダムにサンプルを選択、シャッフルし、インデックスをリセット
    logger.info(f"データを {sample_size} サンプルに制限しました。") #　メッセージをログへ出力

    # 全Sentenceのクリーニング
    df_wrime["Sentence"] = df_wrime["Sentence"].apply(func_clean_text)

    # --------------------------------------------------------------------------------------------
    # 学習,検証データに分割(8:2)
    # --------------------------------------------------------------------------------------------
    split_idx = int(len(df_wrime)*0.8) # 8:2に分割するためのインデックス
    df_train = df_wrime.iloc[:split_idx].reset_index(drop=True) # 学習用データ
    df_val = df_wrime.iloc[split_idx:].reset_index(drop=True) # 検証用データ

    logger.info(f"学習用データサイズ: {df_train.shape}") # メッセージをログへ出力
    logger.info(f"検証用データサイズ: {df_val.shape}") # メッセージをログへ出力

    # --------------------------------------------------------------------------------------------
    # キャッシュ制御
    # --------------------------------------------------------------------------------------------
    cache_dir = "./hf_cache"
    if not args.use_cache:
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
            logger.info("既存キャッシュを削除しました。")

    # --------------------------------------------------------------------------------------------
    # DataFtame の Sentenceをトークン化 & Hugging Face Datasets へ変換する
    # --------------------------------------------------------------------------------------------
    def generate_dataset_from_df(df: pd.DataFrame, tokenizer: AutoTokenizer):
        # DataFrameの各行を辞書に変換したリスト [{"A":1, "B":2}, {"A":3, "B":4}, ...]
        ls_dict_records = df.to_dict(orient="records") 

        # 空のDatasetを作り、mapする
        tmp_dataset = load_dataset(
            "csv",
            data_files={"dummy": file_path},
            split="dummy",
            delimiter="\t",
            cache_dir=cache_dir
        ).from_dict({})  # データを読み込んだ後に、空にする。

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
        for row in ls_dict_records: # rowは各行の辞書
            dict_for_dataset["Sentence"].append(row["Sentence"]) # dataset向け辞書にSentenceを追加していく
            for emo in ls_emotions: 
                dict_for_dataset[emo].append(int(row[emo])) # dataset向け辞書に8感情の強度(正解ラベル)を追加していく

        # dataset化
        dataset_ = tmp_dataset.from_dict(dict_for_dataset)

        # dataset_の各サンプルのSentenceをトークン化
        dataset_ = dataset_.map(
            lambda x: func_tokenize_and_align(x, tokenizer),
            batched=False
        )
        return dataset_ #トークン化後のdatasetを返す

    # --------------------------------------------------------------------------------------------
    # トークナイザ指定 & dataset生成
    # --------------------------------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained("line-corporation/line-distilbert-base-japanese", trust_remote_code=True) # モデルに対応したトークナイザを読み込む
    dataset_train = generate_dataset_from_df(df_train, tokenizer) # 学習用datasetの生成
    dataset_val = generate_dataset_from_df(df_val, tokenizer) # 検証用datasetの生成
    dataset_all = DatasetDict({"train": dataset_train, "validation": dataset_val}) # Hugging Face dataset辞書の作成 

    # --------------------------------------------------------------------------------------------
    # 分類モデルの指定
    #  - ただし CrossEntropyLoss の計算はカスタムTrainerでオーバーライド
    # --------------------------------------------------------------------------------------------
    model = AutoModelForSequenceClassification.from_pretrained(
        "line-corporation/line-distilbert-base-japanese",
        num_labels=32,  # クラス数 = 8感情×4強度
        id2label=dict_id2label, # {クラス:ID} 辞書
        label2id=dict_label2id # {ID:クラス} 辞書
    ).to(device) # モデルをGPU or CPUに転送

    # --------------------------------------------------------------------------------------------
    # 学習,検証時の設定
    # --------------------------------------------------------------------------------------------
    output_dir = "./trainer_output"
    training_args = TrainingArguments(
        output_dir=output_dir, # 学習結果の出力先
        num_train_epochs=num_train_epochs, # Epoch数
        per_device_train_batch_size=100, # 学習時のバッチサイズ
        per_device_eval_batch_size=100, # 検証時のバッチサイズ
        eval_strategy="epoch",  # 検証の頻度
        save_strategy="epoch",  # モデルの保存頻度
        logging_dir="./log",
        logging_steps=100, # 100バッチ毎にログを出力
        load_best_model_at_end=True, # 学習終了後、最良モデルを読み込む
        metric_for_best_model="accuracy_all_emotions", # 最良モデルの指標
        greater_is_better=True, # 正解率が高いほど良い
        seed=42, # シード値
        fp16=True,                                    # メモリ節約のため、半精度学習を有効化
        prediction_loss_only=False                    # 評価メトリクスに 'eval_accuracy_all_emotions'を保持するためFalse
    )

    # --------------------------------------------------------------------------------------------
    # カスタムTrainerを指定 (8感情それぞれについて、4強度のクラス分類を行うため)
    #  - 損失関数がオーバーライドされている。
    # --------------------------------------------------------------------------------------------
    trainer = class_MultiEmotionTrainer(
        model=model, # モデル
        args=training_args, # 学習,検証時の設定
        train_dataset=dataset_all["train"], # 学習用dataset
        eval_dataset=dataset_all["validation"], # 検証用dataset
        tokenizer=tokenizer, # トークナイザ
        data_collator=func_data_collator, # データコレーター
        compute_metrics=func_compute_metrics # 評価関数
    )

    # --------------------------------------------------------------------------------------------
    # epoch=0の場合は、最良モデルをロードする。
    # --------------------------------------------------------------------------------------------
    best_model_path = "./best_model"
    if num_train_epochs == 0:
        logger.info("epoch=0のため、学習は実行されません。最良モデルをロードします。")
        if os.path.exists(best_model_path):
            model = AutoModelForSequenceClassification.from_pretrained("./best_model").to(device)
            logger.info("学習済みの最良モデルをロードしました。")
        else:
            raise FileNotFoundError(f"最良モデルが見つかりません: {best_model_path}") # 保存されたモデルがない場合はエラーを出力して終了
    # ----------------------------------------------------------------------------------------
    # 学習実行
    # ----------------------------------------------------------------------------------------
    else:
        trainer.train()
        trainer.save_model("./best_model") # 学習終了時のモデルを保存。設定で学習終了時に評価関数が最良のモデルをロードするようにしている。
        logger.info("学習完了: 最良モデルを ./best_model に保存しました。")

    # --------------------------------------------------------------------------------------------
    # (学習またはepoch=0後) 学習,検証データ(サンプル削減済)で評価関数を計算する。
    # --------------------------------------------------------------------------------------------
    logger.info("最良モデルを使用し、学習,検証データ(サンプル削減済)で評価関数を計算します。")
    train_metrics = trainer.evaluate(eval_dataset=dataset_all["train"])
    logger.info(f"Train metrics: {train_metrics}")
    val_metrics = trainer.evaluate(eval_dataset=dataset_all["validation"])
    logger.info(f"Val metrics: {val_metrics}")

    # --------------------------------------------------------------------------------------------
    # 元データ(サンプル削減済)に予測結果 32クラス確率(softmax後) & 8感情強度 を追加してCSV出力
    # --------------------------------------------------------------------------------------------
    logger.info("元データ(サンプル削減済)に対して予測し、CSV出力します...")
    model.eval()
    batch_size = 100
    all_logits = []

    for start_idx in range(0, len(df_wrime), batch_size): # バッチサイズ毎にサンプルを処理
        end_idx = start_idx + batch_size # バッチサイズの終わりのインデックス
        sub_df = df_wrime.iloc[start_idx:end_idx] # バッチサイズ分のサンプル
        # トークン化 & モデル予測
        inputs = tokenizer(
            list(sub_df["Sentence"]),
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt", #pytorchテンソルで返す
            return_token_type_ids=False  # ★ DistilBertの場合はtoken_type_idsを受け取らないため
        ).to(device)

        with torch.no_grad(): # 勾配計算を行わないようにする。(予測、評価時に適切)
            outputs = model(**inputs) # モデルに入力を渡して予測
            logits = outputs.logits.cpu().numpy()  # GPU上の予測結果テンソルをCPUへ移動し、numpy形式に変換 shape=(batch_size, 32)
        all_logits.append(logits) # 予測結果をリストに追加
        #all_logits = [
        #array([[...], [...], ...]),  # バッチ1のロジット
        #array([[...], [...], ...]),  # バッチ2のロジット
        #...
        #]
    all_logits = np.concatenate(all_logits, axis=0)  # リストの各numpy配列を結合 (サンプル数, 32)
    #all_logits = array([
        #[...],  # サンプル1
        #[...],  # サンプル2
        #...
    #])

    # 予測値から分類確率,感情強度を計算
    pred_strengths = []
    probs_32 = []
    for i in range(len(df_wrime)): # 1サンプル毎に処理
        row_logits = all_logits[i]  # 1サンプルの予測結果 shape=(32,)
        row_probs = []
        row_strengths = []
        for emo_idx in range(len(ls_emotions)): # 1感情毎に処理
            chunk = row_logits[4*emo_idx : 4*emo_idx + 4] # 1サンプルの予測結果のうち、1感情の4強度
            chunk_softmax = np.exp(chunk) / np.sum(np.exp(chunk)) # 4強度の予測値を合計1になるように、変換
            row_strengths.append(int(np.argmax(chunk_softmax))) # 最大確率の強度を追加
            row_probs.extend(chunk_softmax.tolist())  # extend: リストを分解して追加
        pred_strengths.append(row_strengths) # 1サンプルの8感情のそれぞれの強度を追加 [サンプル1, サンプル2, ...]
        probs_32.append(row_probs) # 1サンプルの32クラスの確率を追加 [サンプル1, サンプル2, ...]

    # 強度を列として追加
    for i, emo in enumerate(ls_emotions): # 列名、列インデックスを取得
        df_wrime[f"Predict_{emo}"] = [pred_strengths[row_idx][i] for row_idx in range(len(df_wrime))] # 行,列を指定して追加

    # 分類確率を列として追加
    arr_probs_32 = np.array(probs_32)  # numpy配列に変換
    for j, label_name in enumerate(ls_labels_32): # 列名、列インデックスを取得
        df_wrime[f"Prob_{label_name}"] = arr_probs_32[:, j] # 列だけを指定して全行を追加

    output_csv = "./predict_wrime-ver1.csv"  # csv保存先
    df_wrime.to_csv(output_csv, index=False, encoding="utf-8-sig")
    logger.info(f"予測結果を保存: {output_csv}")

    logger.info("=== 学習/検証プロセスが完了しました ===")

# ================================================================================================
# main 関数: コマンドライン引数に応じて処理を分岐
# ================================================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=3,
                        help="学習エポック数(0なら学習スキップで評価のみ)")
    parser.add_argument("--use_cache", type=bool, default=True,
                        help="TrueならHugging Faceのキャッシュ使用, Falseならキャッシュ削除")
    parser.add_argument("--seed_data", type=int, default=42,
                        help="データ選択のシード")
    args = parser.parse_args()

    if args.mode == "train":
        func_train_and_eval(args)

if __name__ == "__main__":
    main()
