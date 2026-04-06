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
  } catch (error) {
    target.textContent = `데이터를 불러오지 못했습니다.\n${String(error)}`;
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
