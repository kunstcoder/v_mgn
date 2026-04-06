const DATA_PATHS = {
  input: "data/input_sample.json",
  training: "data/training_sample.json",
  output: "data/output_sample.json",
};

const CONTENT_IDS = {
  input: "input-content",
  training: "training-content",
  output: "output-content",
};

function buildTargetState(outputs = []) {
  return outputs.map((item) => ({
    node_id: item.node_id,
    pressure: Number((item.pressure * 0.98).toFixed(2)),
    velocity: item.velocity.map((value) => Number((value * 0.9).toFixed(2))),
  }));
}

function renderOutputSplitView(outputs = []) {
  const currentTarget = document.getElementById("current-content");
  const targetTarget = document.getElementById("target-content");

  if (!currentTarget || !targetTarget) {
    return;
  }

  if (!Array.isArray(outputs) || outputs.length === 0) {
    currentTarget.textContent = "현재 예측 데이터를 표시할 수 없습니다.";
    targetTarget.textContent = "목표 상태 데이터를 표시할 수 없습니다.";
    return;
  }

  const targetState = buildTargetState(outputs);

  currentTarget.textContent = JSON.stringify(
    {
      tensor_state: "node tensor (예측값)",
      outputs,
    },
    null,
    2,
  );

  targetTarget.textContent = JSON.stringify(
    {
      tensor_state: "target node tensor (비교 기준)",
      target_outputs: targetState,
      delta: outputs.map((current, index) => ({
        node_id: current.node_id,
        pressure_gap: Number((targetState[index].pressure - current.pressure).toFixed(2)),
        velocity_gap: targetState[index].velocity.map((value, axis) =>
          Number((value - current.velocity[axis]).toFixed(2)),
        ),
      })),
    },
    null,
    2,
  );
}

async function loadSample(type) {
  const targetId = CONTENT_IDS[type];
  const target = document.getElementById(targetId);

  if (!target) {
    return;
  }

  target.textContent = "로딩 중...";

  try {
    const response = await fetch(DATA_PATHS[type]);

    if (!response.ok) {
      throw new Error(`요청 실패: ${response.status}`);
    }

    const data = await response.json();
    target.textContent = JSON.stringify(data, null, 2);

    if (type === "output") {
      renderOutputSplitView(data.outputs);
    }
  } catch (error) {
    target.textContent = `데이터를 불러오지 못했습니다.\n${String(error)}`;

    if (type === "output") {
      renderOutputSplitView([]);
    }
  }
}

document.querySelectorAll(".load-btn").forEach((button) => {
  button.addEventListener("click", () => {
    const { target } = button.dataset;

    if (!target || !(target in DATA_PATHS)) {
      return;
    }

    loadSample(target);
  });
});
