"""Mesh/Graph + Training loop 통합 Streamlit 앱."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
import streamlit as st


REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo.data_gen import generate_demo_graph


DEFAULTS = {
    "seed": 7,
    "mesh_type": "triangle",
    "total_steps": 120,
    "learning_rate": 0.12,
    "noise_scale": 0.04,
}

VIEW_DEFAULTS = {
    "node_color_feature": "pred_pressure",
    "edge_color_feature": "pred_edge_value",
    "show_target_graph": True,
}

NODE_FEATURE_OPTIONS = {
    "pred_pressure": "현재 예측 pressure",
    "target_pressure": "목표 pressure",
    "pressure_error": "pressure error",
    "position_error": "position error",
    "displacement": "displacement magnitude",
    "pressure": "초기 pressure",
    "velocity_x": "velocity_x",
    "velocity_y": "velocity_y",
    "velocity_mag": "velocity magnitude",
}

EDGE_FEATURE_OPTIONS = {
    "pred_edge_value": "예측 |Δpressure|",
    "target_edge_value": "목표 |Δpressure|",
    "edge_value_error": "|Δpressure| error",
    "edge_signed_delta": "signed Δpressure",
    "length": "현재 edge length",
    "target_length": "목표 edge length",
    "length_error": "edge length error",
    "relative_dx": "relative_dx",
    "relative_dy": "relative_dy",
}

SEQUENTIAL_COLORSCALE = "YlOrRd"
DIVERGING_COLORSCALE = "RdBu_r"
APP_CSS = """
<style>
    .stApp {
        background: #f7f8fb;
        color: #1b1f2a;
    }
    [data-testid="stHeader"] {
        background: rgba(247, 248, 251, 0.92);
    }
    [data-testid="stSidebar"] > div:first-child {
        background: #eef3ff;
    }
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #d8dde7;
        border-radius: 12px;
        padding: 0.75rem;
    }
</style>
"""

st.set_page_config(page_title="MeshGraphNet Unified Viewer", layout="wide")
st.markdown(APP_CSS, unsafe_allow_html=True)
st.title("MeshGraphNet 통합 뷰어")
st.caption("한 화면에서 mesh 시각화 + 노드/엣지 값 + 학습 수렴 과정을 동시에 확인합니다.")


def _init_state() -> None:
    for key, value in DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value
    for key, value in VIEW_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if "selected_step" not in st.session_state:
        st.session_state["selected_step"] = 0
    if "cached_config" not in st.session_state:
        st.session_state["cached_config"] = None
    if "sim_cache" not in st.session_state:
        st.session_state["sim_cache"] = None


def _restore_defaults() -> None:
    for key, value in DEFAULTS.items():
        st.session_state[key] = value
    for key, value in VIEW_DEFAULTS.items():
        st.session_state[key] = value
    st.session_state["selected_step"] = 0
    st.session_state["cached_config"] = None
    st.session_state["sim_cache"] = None


def _build_training_cache(config: dict[str, int | float | str]) -> dict[str, object]:
    rng = np.random.default_rng(int(config["seed"]))
    graph = generate_demo_graph(seed=int(config["seed"]), mesh_type=str(config["mesh_type"]))

    node_df = pd.DataFrame([asdict(n) for n in graph.node_states])
    edge_df = pd.DataFrame([asdict(e) for e in graph.edge_states])
    node_df["base_x"] = node_df["x"]
    node_df["base_y"] = node_df["y"]
    edge_df["base_relative_dx"] = edge_df["relative_dx"]
    edge_df["base_relative_dy"] = edge_df["relative_dy"]
    edge_df["base_length"] = edge_df["length"]
    node_df["velocity_mag"] = np.hypot(
        node_df["velocity_x"].to_numpy(dtype=float),
        node_df["velocity_y"].to_numpy(dtype=float),
    )

    node_count = len(node_df)
    total_steps = int(config["total_steps"])
    learning_rate = float(config["learning_rate"])
    noise_scale = float(config["noise_scale"])

    init_xy = node_df[["base_x", "base_y"]].to_numpy(dtype=float)
    init_pressure = node_df["pressure"].to_numpy(dtype=float)
    pressure_centered = init_pressure - float(init_pressure.mean())
    target_xy = np.empty_like(init_xy)
    target_xy[:, 0] = (
        init_xy[:, 0]
        + 0.07 * np.sin(2.4 * init_xy[:, 1] + 0.8)
        + 0.045 * node_df["velocity_x"].to_numpy(dtype=float)
        + 0.035 * pressure_centered
    )
    target_xy[:, 1] = (
        init_xy[:, 1]
        + 0.08 * np.cos(2.0 * init_xy[:, 0] - 0.3)
        + 0.045 * node_df["velocity_y"].to_numpy(dtype=float)
        - 0.03 * pressure_centered
    )
    target_pressure = (
        0.65 * np.sin(2.2 * target_xy[:, 0])
        + 0.25 * np.cos(1.3 * target_xy[:, 1])
        + 0.95
    )

    steps = np.arange(total_steps + 1)
    pred_pressure_by_step = np.zeros((total_steps + 1, node_count), dtype=float)
    pred_xy_by_step = np.zeros((total_steps + 1, node_count, 2), dtype=float)

    for step in steps:
        alpha = np.exp(-learning_rate * step)
        pressure_noise = rng.normal(0.0, noise_scale * alpha, size=node_count)
        xy_noise = rng.normal(0.0, max(noise_scale * 0.08 * alpha, 0.0015 * alpha), size=(node_count, 2))
        pred_xy_by_step[step] = alpha * init_xy + (1 - alpha) * target_xy + xy_noise
        pred_pressure_by_step[step] = alpha * init_pressure + (1 - alpha) * target_pressure + pressure_noise

    mse_by_step = np.mean((pred_pressure_by_step - target_pressure[None, :]) ** 2, axis=1)

    edge_src = edge_df["src"].to_numpy(dtype=int)
    edge_dst = edge_df["dst"].to_numpy(dtype=int)

    pred_edge_dx_by_step = pred_xy_by_step[:, edge_dst, 0] - pred_xy_by_step[:, edge_src, 0]
    pred_edge_dy_by_step = pred_xy_by_step[:, edge_dst, 1] - pred_xy_by_step[:, edge_src, 1]
    pred_edge_length_by_step = np.sqrt(pred_edge_dx_by_step**2 + pred_edge_dy_by_step**2)
    target_edge_dx = target_xy[edge_dst, 0] - target_xy[edge_src, 0]
    target_edge_dy = target_xy[edge_dst, 1] - target_xy[edge_src, 1]
    target_edge_length = np.sqrt(target_edge_dx**2 + target_edge_dy**2)

    # 압력차가 줄어드는 과정을 edge value로 관찰
    edge_value_by_step = np.abs(
        pred_pressure_by_step[:, edge_src] - pred_pressure_by_step[:, edge_dst]
    )
    edge_signed_delta_by_step = pred_pressure_by_step[:, edge_src] - pred_pressure_by_step[:, edge_dst]
    target_edge_value = np.abs(target_pressure[edge_src] - target_pressure[edge_dst])
    edge_value_error_by_step = edge_value_by_step - target_edge_value[None, :]

    return {
        "graph": graph,
        "node_df": node_df,
        "edge_df": edge_df,
        "steps": steps,
        "pred_xy_by_step": pred_xy_by_step,
        "target_xy": target_xy,
        "target_pressure": target_pressure,
        "pred_pressure_by_step": pred_pressure_by_step,
        "edge_value_by_step": edge_value_by_step,
        "edge_signed_delta_by_step": edge_signed_delta_by_step,
        "target_edge_value": target_edge_value,
        "edge_value_error_by_step": edge_value_error_by_step,
        "pred_edge_dx_by_step": pred_edge_dx_by_step,
        "pred_edge_dy_by_step": pred_edge_dy_by_step,
        "pred_edge_length_by_step": pred_edge_length_by_step,
        "target_edge_dx": target_edge_dx,
        "target_edge_dy": target_edge_dy,
        "target_edge_length": target_edge_length,
        "mse_by_step": mse_by_step,
    }


def _build_color_scale(values: np.ndarray) -> dict[str, object]:
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        finite_values = np.array([0.0], dtype=float)

    vmin = float(finite_values.min())
    vmax = float(finite_values.max())

    if np.isclose(vmin, vmax):
        delta = max(abs(vmin) * 0.1, 0.1)
        vmin -= delta
        vmax += delta

    if vmin < 0.0 < vmax:
        bound = max(abs(vmin), abs(vmax))
        return {"colorscale": DIVERGING_COLORSCALE, "cmin": -bound, "cmax": bound}

    return {"colorscale": SEQUENTIAL_COLORSCALE, "cmin": vmin, "cmax": vmax}


def _color_for_value(value: float, scale: dict[str, object]) -> str:
    cmin = float(scale["cmin"])
    cmax = float(scale["cmax"])
    span = max(cmax - cmin, 1e-12)
    ratio = min(max((value - cmin) / span, 0.0), 1.0)
    return sample_colorscale(scale["colorscale"], [ratio])[0]


def _render_loss_chart(loss_df: pd.DataFrame, selected_step: int) -> None:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=loss_df["step"],
            y=loss_df["MSE"],
            mode="lines",
            line={"color": "#d97706", "width": 3},
            name="MSE",
        )
    )
    current_point = loss_df.iloc[selected_step]
    fig.add_trace(
        go.Scatter(
            x=[current_point["step"]],
            y=[current_point["MSE"]],
            mode="markers",
            marker={"size": 10, "color": "#1d4ed8", "line": {"width": 1, "color": "#ffffff"}},
            name="현재 step",
        )
    )
    fig.update_layout(
        template="plotly_white",
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        height=260,
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0.0},
    )
    fig.update_xaxes(title="학습 step", gridcolor="rgba(148, 163, 184, 0.18)", zeroline=False)
    fig.update_yaxes(title="MSE", gridcolor="rgba(148, 163, 184, 0.18)", zeroline=False)
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})


def _render_mesh(
    graph_xy: np.ndarray,
    target_xy: np.ndarray,
    node_df: pd.DataFrame,
    edge_df: pd.DataFrame,
    node_feature: str,
    edge_feature: str,
    show_target_graph: bool,
) -> None:
    node_feature_label = NODE_FEATURE_OPTIONS[node_feature]
    edge_feature_label = EDGE_FEATURE_OPTIONS[edge_feature]
    node_values = node_df[node_feature].to_numpy(dtype=float)
    edge_values = edge_df[edge_feature].to_numpy(dtype=float)
    node_scale = _build_color_scale(node_values)
    edge_scale = _build_color_scale(edge_values) if len(edge_df) else _build_color_scale(np.array([0.0]))

    fig = go.Figure()

    if show_target_graph and len(edge_df):
        target_edge_x: list[float | None] = []
        target_edge_y: list[float | None] = []
        for row in edge_df.itertuples(index=False):
            src = int(row.src)
            dst = int(row.dst)
            target_edge_x.extend([float(target_xy[src, 0]), float(target_xy[dst, 0]), None])
            target_edge_y.extend([float(target_xy[src, 1]), float(target_xy[dst, 1]), None])
        fig.add_trace(
            go.Scatter(
                x=target_edge_x,
                y=target_edge_y,
                mode="lines",
                line={"color": "rgba(37, 99, 235, 0.45)", "width": 2, "dash": "dash"},
                hoverinfo="skip",
                name="목표 edges",
                showlegend=True,
            )
        )

    for row in edge_df.itertuples(index=False):
        src = int(row.src)
        dst = int(row.dst)
        selected_value = float(getattr(row, edge_feature))
        hover_lines = [
            f"edge {int(row.edge_id)} ({src} → {dst})",
            f"{edge_feature_label}: {selected_value:.4f}",
            f"예측 |Δpressure|: {float(row.pred_edge_value):.4f}",
            f"목표 |Δpressure|: {float(row.target_edge_value):.4f}",
            f"|Δpressure| error: {float(row.edge_value_error):+.4f}",
            f"signed Δpressure: {float(row.edge_signed_delta):+.4f}",
            f"현재 길이: {float(row.length):.4f}",
            f"목표 길이: {float(row.target_length):.4f}",
            f"길이 오차: {float(row.length_error):+.4f}",
            f"relative_dx: {float(row.relative_dx):+.4f}",
            f"relative_dy: {float(row.relative_dy):+.4f}",
            f"target_dx: {float(row.target_relative_dx):+.4f}",
            f"target_dy: {float(row.target_relative_dy):+.4f}",
        ]
        fig.add_trace(
            go.Scatter(
                x=[graph_xy[src, 0], graph_xy[dst, 0]],
                y=[graph_xy[src, 1], graph_xy[dst, 1]],
                mode="lines",
                line={"color": _color_for_value(selected_value, edge_scale), "width": 4},
                hovertemplate="<br>".join(hover_lines) + "<extra></extra>",
                showlegend=False,
            )
        )

    node_hover = [
        "<br>".join(
            [
                f"node {int(row.node_id)}",
                f"{node_feature_label}: {float(row[node_feature]):.4f}",
                f"x: {float(row['x']):+.4f}",
                f"y: {float(row['y']):+.4f}",
                f"target_x: {float(row['target_x']):+.4f}",
                f"target_y: {float(row['target_y']):+.4f}",
                f"position error: {float(row['position_error']):.4f}",
                f"displacement: {float(row['displacement']):.4f}",
                f"현재 예측 pressure: {float(row['pred_pressure']):.4f}",
                f"목표 pressure: {float(row['target_pressure']):.4f}",
                f"pressure error: {float(row['pressure_error']):+.4f}",
                f"초기 pressure: {float(row['pressure']):.4f}",
                f"velocity_x: {float(row['velocity_x']):+.4f}",
                f"velocity_y: {float(row['velocity_y']):+.4f}",
                f"velocity magnitude: {float(row['velocity_mag']):.4f}",
            ]
        )
        for _, row in node_df.iterrows()
    ]

    fig.add_trace(
        go.Scatter(
            x=graph_xy[:, 0],
            y=graph_xy[:, 1],
            mode="markers",
            marker={
                "size": 16,
                "color": node_values,
                "colorscale": node_scale["colorscale"],
                "cmin": node_scale["cmin"],
                "cmax": node_scale["cmax"],
                "showscale": True,
                "line": {"color": "#0f172a", "width": 1.0},
                "colorbar": {
                    "title": node_feature_label,
                    "x": 1.02,
                    "y": 0.78,
                    "len": 0.46,
                    "thickness": 14,
                },
            },
            text=node_hover,
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        )
    )

    if show_target_graph:
        fig.add_trace(
            go.Scatter(
                x=target_xy[:, 0],
                y=target_xy[:, 1],
                mode="markers",
                marker={
                    "size": 11,
                    "color": "rgba(37, 99, 235, 0.12)",
                    "line": {"color": "rgba(37, 99, 235, 0.8)", "width": 1.5},
                    "symbol": "circle-open",
                },
                hoverinfo="skip",
                name="목표 nodes",
                showlegend=True,
            )
        )

    x_span = float(np.ptp(graph_xy[:, 0]))
    y_span = float(np.ptp(graph_xy[:, 1]))
    x_offset = max(x_span * 0.018, 0.015)
    y_offset = max(y_span * 0.018, 0.015)
    fig.add_trace(
        go.Scatter(
            x=graph_xy[:, 0] + x_offset,
            y=graph_xy[:, 1] + y_offset,
            mode="text",
            text=node_df["node_id"].astype(str),
            textfont={"size": 11, "color": "#111827"},
            hoverinfo="skip",
            showlegend=False,
        )
    )

    if len(edge_df):
        anchor_x = float(graph_xy[0, 0])
        anchor_y = float(graph_xy[0, 1])
        fig.add_trace(
            go.Scatter(
                x=[anchor_x, anchor_x],
                y=[anchor_y, anchor_y],
                mode="markers",
                marker={
                    "size": 0.1,
                    "color": [edge_scale["cmin"], edge_scale["cmax"]],
                    "colorscale": edge_scale["colorscale"],
                    "cmin": edge_scale["cmin"],
                    "cmax": edge_scale["cmax"],
                    "showscale": True,
                    "opacity": 0.0,
                    "colorbar": {
                        "title": edge_feature_label,
                        "x": 1.12,
                        "y": 0.30,
                        "len": 0.46,
                        "thickness": 14,
                    },
                },
                hoverinfo="skip",
                showlegend=False,
            )
        )

    all_xy = graph_xy if not show_target_graph else np.vstack([graph_xy, target_xy])
    x_span = float(np.ptp(all_xy[:, 0]))
    y_span = float(np.ptp(all_xy[:, 1]))
    x_margin = max(x_span * 0.12, 0.08)
    y_margin = max(y_span * 0.12, 0.08)
    fig.update_layout(
        template="plotly_white",
        title=f"Mesh + Graph 상태 (노드 색: {node_feature_label}, 엣지 색: {edge_feature_label})",
        margin={"l": 20, "r": 180, "t": 60, "b": 20},
        height=690,
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        hovermode="closest",
        font={"color": "#1b1f2a"},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.01, "x": 0.0},
    )
    fig.update_xaxes(
        title="x",
        showgrid=True,
        gridcolor="rgba(148, 163, 184, 0.18)",
        zeroline=False,
        range=[float(all_xy[:, 0].min()) - x_margin, float(all_xy[:, 0].max()) + x_margin],
    )
    fig.update_yaxes(
        title="y",
        showgrid=True,
        gridcolor="rgba(148, 163, 184, 0.18)",
        zeroline=False,
        scaleanchor="x",
        scaleratio=1,
        range=[float(all_xy[:, 1].min()) - y_margin, float(all_xy[:, 1].max()) + y_margin],
    )
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})


def _go_next_step(max_step: int) -> None:
    """다음 step으로 이동 (버튼 콜백용)."""
    st.session_state["selected_step"] = min(st.session_state["selected_step"] + 1, max_step)


def _reset_step() -> None:
    """step을 0으로 초기화 (버튼 콜백용)."""
    st.session_state["selected_step"] = 0


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

    st.button("기본값 복원", use_container_width=True, on_click=_restore_defaults)

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
pred_xy_by_step = sim["pred_xy_by_step"]
target_xy = sim["target_xy"]
pred_pressure_by_step = sim["pred_pressure_by_step"]
target_pressure = sim["target_pressure"]
edge_value_by_step = sim["edge_value_by_step"]
edge_signed_delta_by_step = sim["edge_signed_delta_by_step"]
target_edge_value = sim["target_edge_value"]
edge_value_error_by_step = sim["edge_value_error_by_step"]
pred_edge_dx_by_step = sim["pred_edge_dx_by_step"]
pred_edge_dy_by_step = sim["pred_edge_dy_by_step"]
pred_edge_length_by_step = sim["pred_edge_length_by_step"]
target_edge_dx = sim["target_edge_dx"]
target_edge_dy = sim["target_edge_dy"]
target_edge_length = sim["target_edge_length"]
mse_by_step = sim["mse_by_step"]

col_step_a, col_step_b, col_step_c = st.columns([2.1, 1.2, 1.2])
with col_step_a:
    st.slider("학습 step", min_value=0, max_value=int(steps[-1]), key="selected_step")
with col_step_b:
    st.button(
        "다음 step ▶",
        use_container_width=True,
        on_click=_go_next_step,
        args=(int(steps[-1]),),
    )
with col_step_c:
    st.button("step 0으로", use_container_width=True, on_click=_reset_step)

selected_step = int(st.session_state["selected_step"])

node_df["x"] = pred_xy_by_step[selected_step, :, 0]
node_df["y"] = pred_xy_by_step[selected_step, :, 1]
node_df["target_x"] = target_xy[:, 0]
node_df["target_y"] = target_xy[:, 1]
node_df["displacement"] = np.sqrt(
    (node_df["x"] - node_df["base_x"]) ** 2 + (node_df["y"] - node_df["base_y"]) ** 2
)
node_df["position_error"] = np.sqrt(
    (node_df["x"] - node_df["target_x"]) ** 2 + (node_df["y"] - node_df["target_y"]) ** 2
)
node_df["pred_pressure"] = pred_pressure_by_step[selected_step]
node_df["target_pressure"] = target_pressure
node_df["pressure_error"] = node_df["pred_pressure"] - node_df["target_pressure"]
node_df["velocity_mag"] = np.hypot(
    node_df["velocity_x"].to_numpy(dtype=float),
    node_df["velocity_y"].to_numpy(dtype=float),
)

edge_df["pred_edge_value"] = edge_value_by_step[selected_step]
edge_df["target_edge_value"] = target_edge_value
edge_df["edge_value_error"] = edge_value_error_by_step[selected_step]
edge_df["edge_signed_delta"] = edge_signed_delta_by_step[selected_step]
edge_df["relative_dx"] = pred_edge_dx_by_step[selected_step]
edge_df["relative_dy"] = pred_edge_dy_by_step[selected_step]
edge_df["length"] = pred_edge_length_by_step[selected_step]
edge_df["target_relative_dx"] = target_edge_dx
edge_df["target_relative_dy"] = target_edge_dy
edge_df["target_length"] = target_edge_length
edge_df["length_error"] = edge_df["length"] - edge_df["target_length"]

view_col_a, view_col_b, view_col_c, view_col_d = st.columns([1.15, 1.15, 1.0, 1.8])
with view_col_a:
    st.selectbox(
        "노드 컬러 feature",
        options=list(NODE_FEATURE_OPTIONS),
        format_func=lambda key: NODE_FEATURE_OPTIONS[key],
        key="node_color_feature",
    )
with view_col_b:
    st.selectbox(
        "엣지 컬러 feature",
        options=list(EDGE_FEATURE_OPTIONS),
        format_func=lambda key: EDGE_FEATURE_OPTIONS[key],
        key="edge_color_feature",
    )
with view_col_c:
    st.toggle("목표 그래프 오버레이", key="show_target_graph")
with view_col_d:
    st.caption("마우스를 노드나 엣지 위에 올리면 현재 step 기준 상세 feature 값을 팝업으로 볼 수 있습니다.")

st.caption(
    "이 뷰어의 step은 예측 rollout이 target graph로 수렴하는 과정을 단순화해 보여줍니다. "
    "따라서 node 위치와 edge geometry도 step에 따라 함께 변합니다."
)

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
        target_xy=target_xy,
        node_df=node_df,
        edge_df=edge_df,
        node_feature=st.session_state["node_color_feature"],
        edge_feature=st.session_state["edge_color_feature"],
        show_target_graph=bool(st.session_state["show_target_graph"]),
    )

    st.subheader("Loss 곡선")
    _render_loss_chart(loss_df, selected_step)

with main_right:
    st.subheader("현재 step 요약")
    delta = mse_by_step[selected_step] - mse_by_step[selected_step - 1] if selected_step > 0 else 0.0
    st.metric("MSE", f"{mse_by_step[selected_step]:.6f}", delta=f"{delta:+.6f}")
    st.metric("평균 위치 오차", f"{node_df['position_error'].mean():.6f}")
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
                "target_x",
                "target_y",
                "displacement",
                "position_error",
                "pred_pressure",
                "target_pressure",
                "pressure_error",
                "pressure",
                "velocity_x",
                "velocity_y",
                "velocity_mag",
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
                "target_edge_value",
                "edge_value_error",
                "edge_signed_delta",
                "length",
                "target_length",
                "length_error",
                "relative_dx",
                "relative_dy",
                "target_relative_dx",
                "target_relative_dy",
            ]
        ],
        use_container_width=True,
        height=320,
    )

st.success("이제 `streamlit run app.py` 1개만 실행하면 mesh와 학습 step 변화를 동시에 볼 수 있습니다.")
