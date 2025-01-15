# Streamlit関連
import requests
import streamlit as st
import matplotlib
import matplotlib.pyplot as plt

# ================================================================================================
# フロントエンド Streamlitでの簡易UI
# ================================================================================================
#def func_streamlit_app():

st.title("日本語テキストの感情強度予測 (8感情×4強度=32クラス分類)")
st.write("文章を入力して「予測実行」を押すと、8感情の4段階強度を表示します。")

input_text = st.text_area("入力文章", "")
if st.button("予測実行"):
    if not input_text.strip():
        st.warning("文章が空です。")
        #return
    
    # FastAPIエンドポイントを呼び出し
    response = requests.post("http://127.0.0.1:8000/predict", json={"text": input_text})
    result = response.json()

    # 表示
    st.subheader("予測結果 (8感情の強度)")
    for emo, strength in result["emotion_strengths"].items():
        st.write(f"{emo}: {strength}")

    # 棒グラフ描画
    st.subheader("32クラスの確率(棒グラフ)")
    matplotlib.rcParams['font.family'] = 'Meiryo'
    fig, ax = plt.subplots(figsize=(8, 6))
    x_labels = list(result["class_probs"].keys())
    y_vals = list(result["class_probs"].values())
    ax.bar(x_labels, y_vals)
    plt.xticks(rotation=90, fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)