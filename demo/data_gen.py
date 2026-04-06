"""데모용 MeshGraphNet 합성 데이터 생성 모듈.

요구사항:
- 10~30개 노드 규모의 좌표/특성 생성
- 메시(삼각형/사각형) 연결 정보 보관
- 그래프 엣지 인덱스 및 엣지 특성 보관
- 노드/엣지/글로벌 상태를 데이터클래스로 관리
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Literal

import numpy as np


MeshCellType = Literal["triangle", "quad"]


@dataclass
class NodeState:
    """노드 초기 상태."""

    node_id: int
    x: float
    y: float
    velocity_x: float
    velocity_y: float
    pressure: float


@dataclass
class EdgeState:
    """엣지 초기 상태."""

    edge_id: int
    src: int
    dst: int
    relative_dx: float
    relative_dy: float
    length: float


@dataclass
class GlobalState:
    """그래프 전체 초기 상태."""

    viscosity: float
    density: float
    dt: float
    step: int


@dataclass
class DemoGraphData:
    """Streamlit 데모에서 사용하기 쉬운 단일 컨테이너."""

    mesh_type: MeshCellType
    node_states: List[NodeState]
    edge_states: List[EdgeState]
    mesh_cells: np.ndarray  # shape: (n_cells, 3|4)
    edge_index: np.ndarray  # shape: (2, n_edges)
    global_state: GlobalState

    def to_dict(self) -> Dict[str, object]:
        """직렬화/디버깅을 위한 dict 변환."""
        return {
            "mesh_type": self.mesh_type,
            "node_states": [asdict(n) for n in self.node_states],
            "edge_states": [asdict(e) for e in self.edge_states],
            "mesh_cells": self.mesh_cells.tolist(),
            "edge_index": self.edge_index.tolist(),
            "global_state": asdict(self.global_state),
        }


def _sample_node_count(rng: np.random.Generator) -> int:
    """10~30 사이 노드 수를 샘플링."""
    return int(rng.integers(low=10, high=31))


def _grid_dimensions(target_nodes: int) -> tuple[int, int]:
    """target_nodes를 커버하는 2D 격자 (nx, ny) 계산."""
    nx = int(np.ceil(np.sqrt(target_nodes)))
    ny = int(np.ceil(target_nodes / nx))
    return nx, ny


def _build_positions(node_count: int, rng: np.random.Generator) -> np.ndarray:
    """노드 2D 위치 (shape=(node_count, 2)) 생성."""
    nx, ny = _grid_dimensions(node_count)
    xs = np.linspace(0.0, 1.0, nx)
    ys = np.linspace(0.0, 1.0, ny)
    grid = np.array([(x, y) for y in ys for x in xs], dtype=float)
    positions = grid[:node_count].copy()

    # 시각적으로 너무 규칙적이지 않도록 작은 jitter 추가
    jitter = rng.normal(loc=0.0, scale=0.03, size=positions.shape)
    positions += jitter
    return positions


def _build_mesh_cells(nx: int, ny: int, node_count: int, mesh_type: MeshCellType) -> np.ndarray:
    """격자 기반 메시 셀 연결 정보 생성."""

    def node_id(ix: int, iy: int) -> int:
        return iy * nx + ix

    cells: list[list[int]] = []
    for iy in range(ny - 1):
        for ix in range(nx - 1):
            n00 = node_id(ix, iy)
            n10 = node_id(ix + 1, iy)
            n01 = node_id(ix, iy + 1)
            n11 = node_id(ix + 1, iy + 1)

            # 실제 node_count보다 큰 가상 노드 포함 셀은 제거
            if max(n00, n10, n01, n11) >= node_count:
                continue

            if mesh_type == "quad":
                cells.append([n00, n10, n11, n01])
            else:
                cells.append([n00, n10, n11])
                cells.append([n00, n11, n01])

    if not cells:
        width = 4 if mesh_type == "quad" else 3
        return np.empty((0, width), dtype=int)

    return np.asarray(cells, dtype=int)


def _build_edge_index_from_cells(mesh_cells: np.ndarray) -> np.ndarray:
    """셀 연결 정보로부터 방향 그래프 edge_index (2, E) 생성."""
    edge_set: set[tuple[int, int]] = set()
    for cell in mesh_cells:
        k = len(cell)
        for i in range(k):
            a = int(cell[i])
            b = int(cell[(i + 1) % k])
            edge_set.add((a, b))
            edge_set.add((b, a))

    if not edge_set:
        return np.empty((2, 0), dtype=int)

    sorted_edges = sorted(edge_set)
    return np.array(sorted_edges, dtype=int).T


def _build_node_states(positions: np.ndarray, rng: np.random.Generator) -> List[NodeState]:
    """좌표 기반 노드 상태 생성."""
    node_states: list[NodeState] = []
    velocities = rng.normal(0.0, 0.2, size=positions.shape)
    pressure = rng.uniform(0.8, 1.2, size=(positions.shape[0],))

    for i, (x, y) in enumerate(positions):
        node_states.append(
            NodeState(
                node_id=i,
                x=float(x),
                y=float(y),
                velocity_x=float(velocities[i, 0]),
                velocity_y=float(velocities[i, 1]),
                pressure=float(pressure[i]),
            )
        )

    return node_states


def _build_edge_states(edge_index: np.ndarray, positions: np.ndarray) -> List[EdgeState]:
    """edge_index와 좌표로부터 엣지 상태 생성."""
    edge_states: list[EdgeState] = []
    for eid, (src, dst) in enumerate(edge_index.T):
        src_xy = positions[src]
        dst_xy = positions[dst]
        dx_dy = dst_xy - src_xy
        edge_states.append(
            EdgeState(
                edge_id=eid,
                src=int(src),
                dst=int(dst),
                relative_dx=float(dx_dy[0]),
                relative_dy=float(dx_dy[1]),
                length=float(np.linalg.norm(dx_dy)),
            )
        )
    return edge_states


def generate_demo_graph(
    seed: int = 7,
    mesh_type: MeshCellType = "triangle",
) -> DemoGraphData:
    """데모용 합성 그래프/메시 데이터 생성."""
    rng = np.random.default_rng(seed)

    node_count = _sample_node_count(rng)
    positions = _build_positions(node_count=node_count, rng=rng)
    nx, ny = _grid_dimensions(node_count)

    mesh_cells = _build_mesh_cells(nx=nx, ny=ny, node_count=node_count, mesh_type=mesh_type)
    edge_index = _build_edge_index_from_cells(mesh_cells)

    node_states = _build_node_states(positions=positions, rng=rng)
    edge_states = _build_edge_states(edge_index=edge_index, positions=positions)
    global_state = GlobalState(viscosity=0.01, density=1.0, dt=0.02, step=0)

    return DemoGraphData(
        mesh_type=mesh_type,
        node_states=node_states,
        edge_states=edge_states,
        mesh_cells=mesh_cells,
        edge_index=edge_index,
        global_state=global_state,
    )
