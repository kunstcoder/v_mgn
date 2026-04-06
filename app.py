import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(page_title="Streamlit Demo", layout="wide")
st.title("재현 가능한 데모 대시보드")

with st.sidebar:
    st.header("데모 설정")
    show_mesh_graph = st.toggle("메시/그래프", value=True)
    show_init_values = st.toggle("초기값", value=True)
    show_training = st.toggle("학습과정", value=True)
    show_answer = st.toggle("정답 on/off", value=False)
    seed = st.number_input("랜덤 시드", min_value=0, max_value=1_000_000, value=42, step=1)

rng = np.random.default_rng(int(seed))
num_points = 200
x = rng.normal(loc=0.0, scale=1.0, size=num_points)
y = 0.8 * x + rng.normal(loc=0.0, scale=0.5, size=num_points)
loss = np.exp(-np.linspace(0, 5, 60)) + rng.normal(loc=0.0, scale=0.01, size=60)

sample_df = pd.DataFrame({"x": x, "y": y})
loss_df = pd.DataFrame({"epoch": np.arange(1, 61), "loss": loss})

col_left, col_mid, col_right = st.columns([2, 2, 1])

with col_left:
    st.subheader("시각화 패널")
    if show_mesh_graph:
        st.scatter_chart(sample_df, x="x", y="y")
    else:
        st.info("사이드바에서 `메시/그래프`를 켜면 시각화를 표시합니다.")

with col_mid:
    st.subheader("학습/과정 패널")
    if show_training:
        st.line_chart(loss_df, x="epoch", y="loss")
    else:
        st.info("사이드바에서 `학습과정`을 켜면 loss 곡선을 표시합니다.")

with col_right:
    st.subheader("수치 패널")
    if show_init_values:
        st.metric("샘플 수", f"{num_points}")
        st.metric("x 평균", f"{sample_df['x'].mean():.3f}")
        st.metric("y 평균", f"{sample_df['y'].mean():.3f}")
    if show_answer:
        corr = sample_df['x'].corr(sample_df['y'])
        st.success(f"정답(상관계수): {corr:.3f}")
