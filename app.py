# Streamlit関連
import requests
import streamlit as st
import matplotlib
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# ================================================================================================
# FastAPIエンドポイントを呼び出し、リクエストを送信
# ================================================================================================
def send_request_to_api(payload: str):
    try:
        response = requests.post("http://127.0.0.1:8000/predict", json={"text": payload})
        st.write("リクエスト送信中...")
        response.raise_for_status()
        st.write("リクエスト応答成功")
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"リクエスト失敗: {e}")
        return None

# ================================================================================================
# 棒グラフ
# ================================================================================================
def plot_bar_chart1(data: dict) -> go.Figure:
    fig = go.Figure(
        go.Bar(
            y=list(data.keys()),  # 縦軸: 辞書のキー
            x=list(data.values()),  # 横軸: 辞書の値
            orientation="h",  # 横向きの棒グラフ
            marker=dict(color="skyblue"),  # 棒の色
            width=0.2,  # 棒の太さ（デフォルトは 0.8）
        )
    )
    fig.update_layout(
        title="感情強度0~3",
        xaxis_title="強度",
        yaxis_title="感情",
        template="plotly_white",
        width=500,
        height=300,
    )
    fig.update_xaxes(
        range=[0, 3.01],  # 横軸の範囲
        dtick=1,  # 主目盛りの間隔
        tick0=0,  # 主目盛りの開始位置
    )
    return fig

def plot_bar_chart2(data: dict, emo: str) -> go.Figure:
    fig = go.Figure(
        go.Bar(
            y=list(data.keys()),  # 縦軸: 辞書のキー
            x=list(data.values()),  # 横軸: 辞書の値
            orientation="h",  # 横向きの棒グラフ
            marker=dict(color="skyblue"),  # 棒の色
            width=0.2,  # 棒の太さ（デフォルトは 0.8）
        )
    )
    fig.update_layout(
        title=f"{emo}強度分類確率0~1",
        xaxis_title="分類確率",
        yaxis_title="感情強度",
        template="plotly_white",
        width=500,
        height=250,
    )
    fig.update_xaxes(
        range=[0, 1.01],  # 横軸の範囲
        dtick=0.1,  # 主目盛りの間隔
        tick0=0,  # 主目盛りの開始位置
    )
    return fig

# ================================================================================================
# フロントエンド Streamlitでの簡易UI
# ================================================================================================
st.title("日本語テキストの感情強度予測 (8感情×4強度=32クラス分類)")
st.write("文章を入力して「予測実行」を押すと、8感情の4段階強度を表示します。")

input_text = str(st.text_area("入力文章", ""))
if st.button("予測実行"):
    if not input_text.strip():
        st.warning("文章が空です。")
    
    result = send_request_to_api(input_text)

    # 棒グラフを描画
    fig = plot_bar_chart1(result["emotion_strengths"])
    st.plotly_chart(fig)

    # 棒グラフ描画
    ls_emotions = result["emotions"]
    for emo in ls_emotions:
        ls_key = [f"{emo}_0", f"{emo}_1", f"{emo}_2", f"{emo}_3"]
        dict_prob_32 = result["class_probs"]
        fig = plot_bar_chart2({key: dict_prob_32[key] for key in ls_key}, emo)
        st.plotly_chart(fig)