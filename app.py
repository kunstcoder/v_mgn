"""Mesh/Graph + Training loop 통합 Streamlit 앱."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import colors
import numpy as np
import pandas as pd
import streamlit as st


REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo.data_gen import generate_demo_graph


st.set_page_config(page_title="MeshGraphNet Unified Viewer", layout="wide")
st.title("MeshGraphNet 통합 뷰어")
st.caption("한 화면에서 mesh 시각화 + 노드/엣지 값 + step별 학습 수렴 과정을 동시에 확인합니다.")

DEFAULTS = {
    "seed": 7,
    "mesh_type": "triangle",
    "total_steps": 120,
    "learning_rate": 0.12,
    "noise_scale": 0.04,
}


def _init_state() -> None:
    for key, value in DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if "selected_step" not in st.session_state:
        st.session_state["selected_step"] = 0
    if "cached_config" not in st.session_state:
        st.session_state["cached_config"] = None
    if "sim_cache" not in st.session_state:
        st.session_state["sim_cache"] = None


def _build_training_cache(config: dict[str, int | float | str]) -> dict[str, object]:
    rng = np.random.default_rng(int(config["seed"]))
    graph = generate_demo_graph(seed=int(config["seed"]), mesh_type=str(config["mesh_type"]))

    node_df = pd.DataFrame([asdict(n) for n in graph.node_states])
    edge_df = pd.DataFrame([asdict(e) for e in graph.edge_states])

    node_count = len(node_df)
    total_steps = int(config["total_steps"])
    learning_rate = float(config["learning_rate"])
    noise_scale = float(config["noise_scale"])

    init_pressure = node_df["pressure"].to_numpy(dtype=float)
    target_pressure = (
        0.65 * np.sin(2.2 * node_df["x"].to_numpy(dtype=float))
        + 0.25 * np.cos(1.3 * node_df["y"].to_numpy(dtype=float))
        + 0.95
    )

    steps = np.arange(total_steps + 1)
    pred_pressure_by_step = np.zeros((total_steps + 1, node_count), dtype=float)

    for step in steps:
        alpha = np.exp(-learning_rate * step)
        noise = rng.normal(0.0, noise_scale * alpha, size=node_count)
        pred_pressure_by_step[step] = alpha * init_pressure + (1 - alpha) * target_pressure + noise

    mse_by_step = np.mean((pred_pressure_by_step - target_pressure[None, :]) ** 2, axis=1)

    edge_src = edge_df["src"].to_numpy(dtype=int)
    edge_dst = edge_df["dst"].to_numpy(dtype=int)
    base_lengths = edge_df["length"].to_numpy(dtype=float)

    # 압력차가 줄어드는 과정을 edge value로 관찰
    edge_value_by_step = np.abs(
        pred_pressure_by_step[:, edge_src] - pred_pressure_by_step[:, edge_dst]
    )
    edge_length_by_step = base_lengths[None, :] * (1.0 + 0.15 * edge_value_by_step)

    return {
        "graph": graph,
        "node_df": node_df,
        "edge_df": edge_df,
        "steps": steps,
        "target_pressure": target_pressure,
        "pred_pressure_by_step": pred_pressure_by_step,
        "edge_value_by_step": edge_value_by_step,
        "edge_length_by_step": edge_length_by_step,
        "mse_by_step": mse_by_step,
    }


def _render_mesh(graph_xy: np.ndarray, edge_index: np.ndarray, node_values: np.ndarray, edge_values: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6.3))

    segments = [[graph_xy[src], graph_xy[dst]] for src, dst in edge_index.T]
    edge_norm = colors.Normalize(vmin=float(edge_values.min()), vmax=float(edge_values.max()) + 1e-12)
    lc = LineCollection(segments, cmap="plasma", norm=edge_norm, linewidths=1.5, alpha=0.85)
    lc.set_array(edge_values)
    ax.add_collection(lc)

    node_sc = ax.scatter(
        graph_xy[:, 0],
        graph_xy[:, 1],
        c=node_values,
        cmap="viridis",
        s=75,
        edgecolors="black",
        linewidths=0.35,
        zorder=5,
    )

    for node_id, (x, y) in enumerate(graph_xy):
        ax.text(x, y, str(node_id), fontsize=7, color="#111827", ha="left", va="bottom")

    ax.set_title("Mesh + Graph 상태 (노드 색: pressure, 엣지 색: |Δpressure|)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(alpha=0.2)
    ax.set_aspect("equal", adjustable="datalim")

    fig.colorbar(node_sc, ax=ax, pad=0.01, label="node pressure")
    fig.colorbar(lc, ax=ax, pad=0.10, label="edge |Δpressure|")
    st.pyplot(fig, use_container_width=True)


_init_state()

with st.sidebar:
    st.subheader("통합 실행 설정")
    st.number_input("seed", min_value=0, max_value=999_999, step=1, key="seed")
    st.radio("mesh type", ["triangle", "quad"], key="mesh_type", horizontal=True)
    st.number_input("total steps", min_value=10, max_value=600, step=10, key="total_steps")
    st.number_input(
        "learning rate", min_value=0.001, max_value=1.0, step=0.001, format="%.3f", key="learning_rate"
    )
    st.slider("noise scale", min_value=0.0, max_value=0.3, step=0.01, key="noise_scale")

    if st.button("기본값 복원", use_container_width=True):
        for key, value in DEFAULTS.items():
            st.session_state[key] = value
        st.session_state["selected_step"] = 0
        st.session_state["cached_config"] = None
        st.session_state["sim_cache"] = None
        st.rerun()

config = {
    "seed": int(st.session_state["seed"]),
    "mesh_type": str(st.session_state["mesh_type"]),
    "total_steps": int(st.session_state["total_steps"]),
    "learning_rate": float(st.session_state["learning_rate"]),
    "noise_scale": float(st.session_state["noise_scale"]),
}

if st.session_state["cached_config"] != config:
    st.session_state["sim_cache"] = _build_training_cache(config)
    st.session_state["cached_config"] = config.copy()
    st.session_state["selected_step"] = 0

sim = st.session_state["sim_cache"]
graph = sim["graph"]
node_df = sim["node_df"].copy()
edge_df = sim["edge_df"].copy()
steps = sim["steps"]
pred_pressure_by_step = sim["pred_pressure_by_step"]
target_pressure = sim["target_pressure"]
edge_value_by_step = sim["edge_value_by_step"]
edge_length_by_step = sim["edge_length_by_step"]
mse_by_step = sim["mse_by_step"]

col_step_a, col_step_b, col_step_c = st.columns([2.1, 1.2, 1.2])
with col_step_a:
    st.slider("학습 step", min_value=0, max_value=int(steps[-1]), key="selected_step")
with col_step_b:
    if st.button("다음 step ▶", use_container_width=True):
        st.session_state["selected_step"] = min(st.session_state["selected_step"] + 1, int(steps[-1]))
        st.rerun()
with col_step_c:
    if st.button("step 0으로", use_container_width=True):
        st.session_state["selected_step"] = 0
        st.rerun()

selected_step = int(st.session_state["selected_step"])

node_df["pred_pressure"] = pred_pressure_by_step[selected_step]
node_df["target_pressure"] = target_pressure
node_df["pressure_error"] = node_df["pred_pressure"] - node_df["target_pressure"]

edge_df["pred_edge_value"] = edge_value_by_step[selected_step]
edge_df["pred_length"] = edge_length_by_step[selected_step]

loss_df = pd.DataFrame({"step": steps, "MSE": mse_by_step})

st.info(
    f"노드 {len(node_df)}개 / 엣지 {len(edge_df)}개 | "
    f"현재 step={selected_step}, MSE={mse_by_step[selected_step]:.6f}"
)

main_left, main_right = st.columns([1.6, 1.0])
with main_left:
    xy = node_df[["x", "y"]].to_numpy(dtype=float)
    _render_mesh(
        graph_xy=xy,
        edge_index=graph.edge_index,
        node_values=node_df["pred_pressure"].to_numpy(dtype=float),
        edge_values=edge_df["pred_edge_value"].to_numpy(dtype=float),
    )

    st.subheader("Loss 곡선")
    st.line_chart(loss_df, x="step", y="MSE")

with main_right:
    st.subheader("현재 step 요약")
    delta = mse_by_step[selected_step] - mse_by_step[selected_step - 1] if selected_step > 0 else 0.0
    st.metric("MSE", f"{mse_by_step[selected_step]:.6f}", delta=f"{delta:+.6f}")
    st.metric("평균 |노드 오류|", f"{node_df['pressure_error'].abs().mean():.6f}")
    st.metric("평균 엣지 값 |Δp|", f"{edge_df['pred_edge_value'].mean():.6f}")

st.subheader("노드/엣지 값 테이블 (현재 step)")
table_left, table_right = st.columns([1, 1])
with table_left:
    st.caption("Node states")
    st.dataframe(
        node_df[
            [
                "node_id",
                "x",
                "y",
                "pred_pressure",
                "target_pressure",
                "pressure_error",
                "velocity_x",
                "velocity_y",
            ]
        ],
        use_container_width=True,
        height=320,
    )

with table_right:
    st.caption("Edge states")
    st.dataframe(
        edge_df[
            [
                "edge_id",
                "src",
                "dst",
                "pred_edge_value",
                "pred_length",
                "length",
                "relative_dx",
                "relative_dy",
            ]
        ],
        use_container_width=True,
        height=320,
    )

st.success("이제 `streamlit run app.py` 1개만 실행하면 mesh와 학습 step 변화를 동시에 볼 수 있습니다.")
