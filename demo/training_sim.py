import numpy as np
import pandas as pd
import streamlit as st
from matplotlib import colors
import matplotlib.pyplot as plt


st.set_page_config(page_title="Training Progress Simulator", layout="wide")
st.title("학습 진행 데모 시뮬레이터")
st.caption("step별 예측값 수렴, 손실 곡선, 메트릭을 한 번에 확인합니다.")

with st.sidebar:
    st.header("시뮬레이션 설정")
    total_steps = st.slider("총 스텝 수 (N)", min_value=20, max_value=400, value=120, step=10)
    num_nodes = st.slider("노드 수", min_value=10, max_value=300, value=80, step=10)
    noise_scale = st.slider("노이즈 강도", min_value=0.0, max_value=0.5, value=0.08, step=0.01)
    seed = st.number_input("랜덤 시드", min_value=0, max_value=1_000_000, value=42, step=1)

rng = np.random.default_rng(int(seed))
node_ids = np.arange(num_nodes)

# 목표값(target): 완만한 비선형 곡선
x = np.linspace(-2.5, 2.5, num_nodes)
target = 0.7 * np.sin(1.4 * x) + 0.3 * np.tanh(x)

# 초기 예측값: 타깃과 차이가 있는 랜덤 상태
initial_pred = rng.normal(loc=0.0, scale=1.0, size=num_nodes)

steps = np.arange(total_steps + 1)
preds_by_step = np.zeros((total_steps + 1, num_nodes))

for step in steps:
    # step이 커질수록 (1 - alpha)가 작아져 target으로 수렴
    alpha = np.exp(-step / max(total_steps * 0.22, 1.0))
    step_noise = rng.normal(loc=0.0, scale=noise_scale * alpha, size=num_nodes)
    preds_by_step[step] = alpha * initial_pred + (1 - alpha) * target + step_noise

errors = preds_by_step - target
mse_by_step = np.mean(errors**2, axis=1)
mae_by_step = np.mean(np.abs(errors), axis=1)

loss_df = pd.DataFrame(
    {
        "step": steps,
        "loss (MSE)": mse_by_step,
        "MAE": mae_by_step,
    }
)

selected_step = st.slider("step 이동", min_value=0, max_value=total_steps, value=0, step=1)
show_target_overlay = st.checkbox("정답 데이터 표시", value=True)

current_pred = preds_by_step[selected_step]
node_df = pd.DataFrame(
    {
        "node": node_ids,
        "target": target,
        "prediction": current_pred,
        "error": current_pred - target,
    }
)

col1, col2 = st.columns([2.5, 1.3])

with col1:
    st.subheader("현재 step의 노드 상태")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(
        node_df["node"],
        node_df["prediction"],
        color="#2563eb",
        marker="o",
        markersize=4,
        linewidth=1.7,
        label="prediction",
    )

    if show_target_overlay:
        ax.plot(
            node_df["node"],
            node_df["target"],
            color="#f97316",
            marker="x",
            markersize=5,
            linewidth=1.4,
            linestyle="--",
            label="target",
        )

    y_min = min(np.min(preds_by_step), np.min(target))
    y_max = max(np.max(preds_by_step), np.max(target))
    y_pad = 0.1 * (y_max - y_min + 1e-9)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.set_xlabel("node")
    ax.set_ylabel("value")
    ax.set_title(f"Node-wise prediction (step={selected_step})")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")
    st.pyplot(fig, use_container_width=True)

    st.caption("슬라이더를 움직이면 해당 시점의 prediction이 즉시 갱신됩니다.")

    st.subheader("손실 곡선 (loss vs step)")
    st.line_chart(loss_df, x="step", y="loss (MSE)")

with col2:
    st.subheader(f"step {selected_step} 요약 메트릭")

    delta_mse = (
        mse_by_step[selected_step] - mse_by_step[selected_step - 1]
        if selected_step > 0
        else 0.0
    )
    delta_mae = (
        mae_by_step[selected_step] - mae_by_step[selected_step - 1]
        if selected_step > 0
        else 0.0
    )

    st.metric("MSE", f"{mse_by_step[selected_step]:.6f}", delta=f"{delta_mse:+.6f}")
    st.metric("MAE", f"{mae_by_step[selected_step]:.6f}", delta=f"{delta_mae:+.6f}")
    st.metric(
        "RMSE",
        f"{np.sqrt(mse_by_step[selected_step]):.6f}",
    )

    st.write("### 현재 step 오류 분포")
    err_abs_max = float(np.max(np.abs(errors)))
    err_norm = colors.TwoSlopeNorm(vmin=-err_abs_max, vcenter=0.0, vmax=err_abs_max)
    err_cmap = plt.get_cmap("RdBu_r")
    bar_colors = err_cmap(err_norm(node_df["error"].to_numpy()))

    fig_err, ax_err = plt.subplots(figsize=(6, 3.8))
    ax_err.bar(node_df["node"], node_df["error"], color=bar_colors, width=0.8)
    ax_err.axhline(0.0, color="#111827", linewidth=1.0)
    ax_err.set_xlabel("node")
    ax_err.set_ylabel("prediction - target")
    ax_err.set_title("Error by node")
    ax_err.set_ylim(-err_abs_max * 1.1, err_abs_max * 1.1)
    ax_err.grid(axis="y", alpha=0.25)

    sm = plt.cm.ScalarMappable(norm=err_norm, cmap=err_cmap)
    sm.set_array([])
    fig_err.colorbar(sm, ax=ax_err, pad=0.02, label="error scale (fixed)")
    st.pyplot(fig_err, use_container_width=True)
