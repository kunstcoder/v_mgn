"""합성 MeshGraphNet 데이터 Streamlit 시각화 데모."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# `streamlit run demo/streamlit_app.py` 실행 시 작업 디렉터리에 따라
# `demo` 패키지 해석이 실패할 수 있어 루트 경로를 명시적으로 보강한다.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo.data_gen import generate_demo_graph


st.set_page_config(page_title="MeshGraphNet Demo Data", layout="wide")
st.title("MeshGraphNet Demo: 합성 데이터 생성 및 시각화")

with st.sidebar:
    st.subheader("생성 옵션")
    seed = st.number_input("Seed", min_value=0, max_value=99999, value=7, step=1)
    mesh_type = st.radio("메시 타입", options=["triangle", "quad"], horizontal=True)


graph = generate_demo_graph(seed=int(seed), mesh_type=mesh_type)

node_df = pd.DataFrame([asdict(n) for n in graph.node_states])
edge_df = pd.DataFrame([asdict(e) for e in graph.edge_states])
global_df = pd.DataFrame([asdict(graph.global_state)])
mesh_df = pd.DataFrame(graph.mesh_cells)
mesh_df.columns = [f"v{i}" for i in range(mesh_df.shape[1])]

left_col, right_col = st.columns([1, 1])

with left_col:
    st.subheader("초기 상태 테이블")
    st.caption("노드 / 엣지 / 글로벌 상태")
    st.markdown("**Node features**")
    st.dataframe(node_df, use_container_width=True, height=260)

    st.markdown("**Edge features**")
    st.dataframe(edge_df, use_container_width=True, height=260)

    st.markdown("**Global state**")
    st.dataframe(global_df, use_container_width=True)

    st.markdown("**Mesh connectivity**")
    st.dataframe(mesh_df, use_container_width=True, height=220)

with right_col:
    st.subheader("2D 그래프 플롯")
    fig, ax = plt.subplots(figsize=(7, 6))

    # 엣지 선분
    xy = node_df[["x", "y"]].to_numpy()
    for src, dst in graph.edge_index.T:
        p0 = xy[src]
        p1 = xy[dst]
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color="#94a3b8", linewidth=1.0, zorder=1)

    # 노드 산점도
    sc = ax.scatter(
        node_df["x"],
        node_df["y"],
        c=node_df["pressure"],
        cmap="viridis",
        s=60,
        edgecolors="black",
        linewidths=0.3,
        zorder=2,
    )

    for _, row in node_df.iterrows():
        ax.text(row["x"], row["y"], f"{int(row['node_id'])}", fontsize=8, ha="left", va="bottom")

    fig.colorbar(sc, ax=ax, label="pressure")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Nodes + Edges ({graph.mesh_type}, N={len(graph.node_states)})")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.2)

    st.pyplot(fig, use_container_width=True)

st.success(
    f"생성 완료: 노드 {len(graph.node_states)}개 / 엣지 {graph.edge_index.shape[1]}개 / 셀 {graph.mesh_cells.shape[0]}개"
)
