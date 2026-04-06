import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib import colors


st.set_page_config(page_title="Training Progress Simulator", layout="wide")
st.title("학습 진행 데모 시뮬레이터")
st.caption("step별 예측값 수렴, 손실 곡선, 메트릭을 한 번에 확인합니다.")

DEFAULT_CONFIG = {
    "num_nodes": 80,
    "edge_density": 0.14,
    "learning_rate": 0.12,
    "total_steps": 120,
    "noise_scale": 0.08,
    "seed": 42,
}


def _init_session_state() -> None:
    for key, value in DEFAULT_CONFIG.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if "sim_config" not in st.session_state:
        st.session_state["sim_config"] = None
    if "sim_data" not in st.session_state:
        st.session_state["sim_data"] = None
    if "selected_step" not in st.session_state:
        st.session_state["selected_step"] = 0


def _validate_config(config: dict[str, float | int]) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    node_count = int(config["num_nodes"])
    edge_density = float(config["edge_density"])
    total_steps = int(config["total_steps"])
    learning_rate = float(config["learning_rate"])

    if edge_density <= 0.0 or edge_density > 1.0:
        errors.append("엣지 밀도는 0보다 크고 1 이하여야 합니다.")

    max_undirected_edges = node_count * (node_count - 1) // 2
    expected_edges = int(max_undirected_edges * edge_density)

    if expected_edges < node_count - 1:
        warnings.append(
            "엣지 수가 매우 적어 그래프가 희소합니다. 수렴 속도가 비정상적으로 느릴 수 있습니다."
        )

    if expected_edges > 5_000:
        warnings.append(
            "엣지 밀도가 높아 예상 엣지 수가 큽니다. 브라우저 렌더링/실험 반복 속도가 느려질 수 있습니다."
        )

    if total_steps < 10:
        errors.append("스텝 수는 최소 10 이상이어야 합니다.")

    if learning_rate <= 0.0:
        errors.append("학습률은 0보다 커야 합니다.")

    if learning_rate > 1.0:
        warnings.append("학습률이 매우 큽니다. 손실 곡선이 급격하게 변할 수 있습니다.")

    return errors, warnings


def _generate_simulation_data(config: dict[str, float | int]) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(int(config["seed"]))
    num_nodes = int(config["num_nodes"])
    total_steps = int(config["total_steps"])
    noise_scale = float(config["noise_scale"])
    learning_rate = float(config["learning_rate"])

    node_ids = np.arange(num_nodes)

    x = np.linspace(-2.5, 2.5, num_nodes)
    target = 0.7 * np.sin(1.4 * x) + 0.3 * np.tanh(x)

    initial_pred = rng.normal(loc=0.0, scale=1.0, size=num_nodes)

    steps = np.arange(total_steps + 1)
    preds_by_step = np.zeros((total_steps + 1, num_nodes))

    for step in steps:
        alpha = np.exp(-learning_rate * step)
        step_noise = rng.normal(loc=0.0, scale=noise_scale * alpha, size=num_nodes)
        preds_by_step[step] = alpha * initial_pred + (1 - alpha) * target + step_noise

    errors = preds_by_step - target
    mse_by_step = np.mean(errors**2, axis=1)
    mae_by_step = np.mean(np.abs(errors), axis=1)

    return {
        "node_ids": node_ids,
        "target": target,
        "steps": steps,
        "preds_by_step": preds_by_step,
        "errors": errors,
        "mse_by_step": mse_by_step,
        "mae_by_step": mae_by_step,
    }


_init_session_state()

with st.sidebar:
    st.header("시뮬레이션 설정")

    st.number_input("노드 수", min_value=10, max_value=300, step=10, key="num_nodes")
    st.number_input(
        "엣지 밀도 (0~1)",
        min_value=0.01,
        max_value=1.2,
        step=0.01,
        format="%.2f",
        key="edge_density",
        help="밀도가 높을수록 노드 간 연결이 많아집니다.",
    )
    st.number_input(
        "학습률",
        min_value=0.001,
        max_value=1.5,
        step=0.001,
        format="%.3f",
        key="learning_rate",
    )
    st.number_input("총 스텝 수 (N)", min_value=10, max_value=500, step=10, key="total_steps")
    st.slider("노이즈 강도", min_value=0.0, max_value=0.5, step=0.01, key="noise_scale")
    st.number_input("랜덤 시드", min_value=0, max_value=1_000_000, step=1, key="seed")

    if st.button("기본값 복원", use_container_width=True):
        for k, v in DEFAULT_CONFIG.items():
            st.session_state[k] = v
        st.session_state["sim_config"] = None
        st.session_state["sim_data"] = None
        st.rerun()

    export_payload = {
        k: (round(v, 6) if isinstance(v, float) else v)
        for k, v in {
            "num_nodes": st.session_state["num_nodes"],
            "edge_density": st.session_state["edge_density"],
            "learning_rate": st.session_state["learning_rate"],
            "total_steps": st.session_state["total_steps"],
            "noise_scale": st.session_state["noise_scale"],
            "seed": st.session_state["seed"],
        }.items()
    }
    st.download_button(
        label="현재 설정 JSON 내보내기",
        data=json.dumps(export_payload, ensure_ascii=False, indent=2),
        file_name="simulation_config.json",
        mime="application/json",
        use_container_width=True,
    )

current_config = {
    "num_nodes": int(st.session_state["num_nodes"]),
    "edge_density": float(st.session_state["edge_density"]),
    "learning_rate": float(st.session_state["learning_rate"]),
    "total_steps": int(st.session_state["total_steps"]),
    "noise_scale": float(st.session_state["noise_scale"]),
    "seed": int(st.session_state["seed"]),
}

errors, warnings = _validate_config(current_config)
for warning_msg in warnings:
    st.warning(warning_msg)

if errors:
    for err in errors:
        st.error(err)
    st.stop()

if st.session_state["sim_config"] != current_config:
    st.session_state["sim_data"] = _generate_simulation_data(current_config)
    st.session_state["sim_config"] = current_config.copy()

sim_data = st.session_state["sim_data"]

node_ids = sim_data["node_ids"]
target = sim_data["target"]
steps = sim_data["steps"]
preds_by_step = sim_data["preds_by_step"]
errors = sim_data["errors"]
mse_by_step = sim_data["mse_by_step"]
mae_by_step = sim_data["mae_by_step"]

max_undirected_edges = current_config["num_nodes"] * (current_config["num_nodes"] - 1) // 2
estimated_edges = int(max_undirected_edges * current_config["edge_density"])

st.info(
    f"현재 구성: 노드 {current_config['num_nodes']}개, 예상 엣지 {estimated_edges:,}개, "
    f"학습률 {current_config['learning_rate']:.3f}, 스텝 {current_config['total_steps']}"
)

loss_df = pd.DataFrame(
    {
        "step": steps,
        "loss (MSE)": mse_by_step,
        "MAE": mae_by_step,
    }
)

selected_step = st.slider(
    "step 이동",
    min_value=0,
    max_value=current_config["total_steps"],
    value=int(st.session_state["selected_step"]),
    step=1,
    key="selected_step",
)
show_target_overlay = st.checkbox("정답 데이터 표시", value=True)
enable_visualization = st.toggle("시각화 켜기", value=True)

loop_col1, loop_col2, loop_col3 = st.columns([1.2, 1.2, 3.0])
with loop_col1:
    if st.button("다음 step ▶", use_container_width=True):
        st.session_state["selected_step"] = min(
            st.session_state["selected_step"] + 1, current_config["total_steps"]
        )
        st.rerun()
with loop_col2:
    if st.button("step 초기화", use_container_width=True):
        st.session_state["selected_step"] = 0
        st.rerun()
with loop_col3:
    jump_step = st.slider(
        "루프 점프(step)", min_value=1, max_value=min(30, current_config["total_steps"]), value=5, step=1
    )
    if st.button("점프 적용", use_container_width=True):
        st.session_state["selected_step"] = min(
            st.session_state["selected_step"] + jump_step, current_config["total_steps"]
        )
        st.rerun()

selected_step = int(st.session_state["selected_step"])

current_pred = preds_by_step[selected_step]
node_df = pd.DataFrame(
    {
        "node": node_ids,
        "target": target,
        "prediction": current_pred,
        "error": current_pred - target,
    }
)

if not enable_visualization:
    st.warning("시각화가 꺼져 있습니다. 토글을 켜면 step 변화 그래프를 다시 볼 수 있습니다.")
    st.dataframe(node_df, use_container_width=True, height=300)
    st.line_chart(loss_df, x="step", y=["loss (MSE)", "MAE"])
    st.stop()

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

    st.subheader("학습 루프 변화 추적")
    track_count = st.slider("추적할 노드 수", min_value=1, max_value=6, value=3, step=1)
    tracked_nodes = node_ids[:track_count]
    trajectory_df = pd.DataFrame({"step": steps})
    for node_id in tracked_nodes:
        trajectory_df[f"node_{node_id}"] = preds_by_step[:, node_id]
    st.line_chart(trajectory_df, x="step", y=[f"node_{node_id}" for node_id in tracked_nodes])
    st.caption("선택한 노드들의 예측값이 step 진행에 따라 어떻게 수렴하는지 보여줍니다.")

with col2:
    st.subheader(f"step {selected_step} 요약 메트릭")

    delta_mse = (
        mse_by_step[selected_step] - mse_by_step[selected_step - 1] if selected_step > 0 else 0.0
    )
    delta_mae = (
        mae_by_step[selected_step] - mae_by_step[selected_step - 1] if selected_step > 0 else 0.0
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
