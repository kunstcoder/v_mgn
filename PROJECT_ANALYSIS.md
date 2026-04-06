# v_mgn 프로젝트 분석

## 개요

이 저장소는 MeshGraphNet(MGN)을 이해하고 바로 시연해 볼 수 있도록 만든 학습용 데모 프로젝트다. 현재 상태는 크게 두 축으로 구성된다.

- 정적 웹 데모: 샘플 JSON을 불러와 입력, 학습 로그, 출력 데이터를 보여준다.
- Streamlit 데모: 합성 메시/그래프 데이터를 만들고 학습 step 변화까지 시각화한다.

최근 커밋 흐름을 보면, 단순한 정적 데모에서 시작해 합성 데이터 생성기와 Streamlit 기반 시뮬레이터를 추가했고, 마지막에는 설명용 웹페이지와 README를 보강하는 방향으로 확장되었다.

## 현재 코드 구조

### 정적 웹 영역

- `index.html`
  - 메인 정적 데모 화면이다.
  - Input / Training / Output 샘플 데이터를 버튼으로 로드한다.
  - Output 영역에서는 현재 예측값과 목표 상태를 나란히 비교하는 split view를 제공한다.
- `mgn_easy.html`
  - MGN 개념을 쉬운 언어로 설명하는 문서형 웹페이지다.
  - Mesh -> Graph 변환, message passing, PINN과의 관계, 비유 설명을 담고 있다.
- `app.js`
  - `data/*.json`을 fetch해서 화면에 렌더링한다.
  - 출력 데이터에 대해 target state와 delta를 가공해 비교 뷰를 만든다.
- `styles.css`
  - 정적 페이지 공통 스타일이다.
  - flow diagram, split view, 비교 테이블, 반응형 레이아웃까지 포함한다.
- `serve.py`
  - 정적 파일 서버다.
  - `/`는 `index.html`, `/mgn-easy`는 `mgn_easy.html`로 라우팅한다.

### Streamlit 영역

- `demo/data_gen.py`
  - 프로젝트의 핵심 데이터 생성 모듈이다.
  - 합성 노드 좌표, 메시 셀, 방향 그래프 엣지, 노드/엣지/글로벌 상태를 생성한다.
  - triangle / quad 메시를 모두 지원한다.
- `demo/streamlit_app.py`
  - 합성 그래프 자체를 확인하는 기본 Streamlit 시각화 앱이다.
  - 노드/엣지/글로벌 상태 테이블과 2D 그래프 플롯을 제공한다.
- `demo/training_sim.py`
  - 학습 수렴 과정을 step 단위로 보여주는 시뮬레이터다.
  - 예측값, target overlay, loss 곡선, MAE/MSE/RMSE, 오류 분포, 노드 trajectory 추적 기능을 제공한다.
- `app.py`
  - 현재 메인 Streamlit 앱이다.
  - 메시 시각화와 학습 시뮬레이션을 하나로 통합했다.
  - 사이드바 설정, step 슬라이더, 다음 step 버튼, loss 차트, 노드/엣지 테이블을 한 화면에서 제공한다.

### 데이터 영역

- `data/input_sample.json`
  - 정적 입력 예시 데이터
- `data/training_sample.json`
  - 정적 학습 과정 예시 데이터
- `data/output_sample.json`
  - 정적 출력 예시 데이터

## 현재 앱이 실제로 하는 일

### 1. MGN 구조를 설명한다

정적 웹페이지와 쉬운 설명 페이지를 통해 MGN의 핵심 개념을 문서형으로 전달한다. 이 부분은 설명과 예시 중심이며 실제 학습 엔진은 아니다.

### 2. 합성 메시/그래프 데이터를 만든다

`demo/data_gen.py`는 임의의 노드 수를 샘플링하고, 이를 격자 기반 위치와 메시 셀로 변환한다. 이후 셀 연결을 바탕으로 edge index를 만들고, 노드의 속도/압력과 엣지 길이 같은 상태값을 합성한다.

### 3. 학습 수렴 과정을 흉내 낸다

`demo/training_sim.py`와 `app.py`는 실제 MeshGraphNet 모델을 학습시키지 않는다. 대신 초기 예측값이 target 쪽으로 지수적으로 수렴하는 synthetic process를 만들어, step이 진행될수록 MSE가 줄어드는 흐름을 시각화한다.

### 4. mesh와 training 관점을 통합해서 보여준다

최종 메인 앱인 `app.py`는 노드 pressure, 엣지 pressure 차이, loss 곡선, step별 메트릭, 노드/엣지 테이블을 동시에 보여준다. 즉, "그래프 상태"와 "학습 진행"을 한 화면에서 같이 보는 구조다.

## 커밋 히스토리 분석

아래는 `main` 기준 최근 히스토리를 단계별로 재구성한 내용이다.

### 1단계: 저장소 뼈대와 정적 데모 시작

- `73874f5` Initialize repository
  - 저장소 초기화
- `6ee72c4` Add minimal static MeshGraphNet visualization demo scaffold
  - HTML/CSS/JS 기반 정적 데모 골격 추가
  - 샘플 JSON을 보여주는 가장 초기 형태의 데모 구축

### 2단계: Streamlit 진입

- `ff6dd68` Add reproducible Streamlit entrypoint layout
  - Streamlit 실행 진입점을 정리
- `038c056` Add synthetic demo graph generator and Streamlit visualizer
  - 합성 그래프 생성기와 기본 시각화 앱 추가
  - 현재 `demo/data_gen.py`, `demo/streamlit_app.py` 계열의 기반이 이 단계에서 들어옴

### 3단계: 학습 과정 시뮬레이션 추가

- `484a8e7` Add Streamlit training progress simulator demo
  - 별도 training simulator 도입
- `c690a83` Add target overlay and fixed error colormap in training viz
  - 정답 오버레이와 고정 오류 색상 스케일 추가
- `113340c` Expand sidebar config and session-state driven simulation
  - 시뮬레이션 설정을 확장하고 session state 기반으로 재구성
- `e0549ab` Enhance explanatory UI with pipeline flow and split comparison
  - 설명 UI를 강화하고 비교 중심의 정적 화면을 개선
- `16ae2f2` Fix demo module resolution for Streamlit entrypoint
  - `streamlit run demo/streamlit_app.py` 시 import 경로 문제 수정
- `aa1e84c` feat: add training loop controls and visualization toggle
  - 다음 step, 점프, 시각화 on/off 같은 상호작용 기능 추가

### 4단계: 통합 뷰어 완성

- `0f12f25` Integrate mesh app and training simulator into unified Streamlit viewer
  - 메시 시각화와 학습 시뮬레이션을 `app.py` 하나로 통합
  - 현재 프로젝트의 중심 앱이 이 커밋에서 완성됨
- `ba56204` Fix step navigation callback crash in unified Streamlit app
  - step 이동 콜백 관련 충돌 수정

### 5단계: 문서화 및 교육용 페이지 강화

- `205e33f` docs: expand README and add easy MGN web page
  - `README.md` 확장
  - `mgn_easy.html`, `serve.py`, 스타일 보강 추가
  - 이 저장소를 "실행 가능한 데모 + 설명 자료" 형태로 정리

## 현재 상태에 대한 해석

이 프로젝트는 실제 MGN 학습 프레임워크라기보다는, 아래 목적에 초점을 둔 교육/데모 저장소다.

- MGN 개념을 시각적으로 설명
- 메시와 그래프 구조를 직관적으로 보여줌
- 학습 step에 따른 수렴 패턴을 synthetic data로 시연
- 정적 웹과 Streamlit 두 방식으로 접근성 제공

즉, "물리 기반 그래프 모델을 빠르게 설명하고 보여주는 데모 패키지"라고 보는 것이 가장 정확하다.

## 이번에 추가한 항목

- `requirements.txt`
  - 현재 코드에서 실제 사용하는 Python 런타임 패키지를 고정 버전으로 정리했다.
- `.venv`
  - 로컬 가상환경을 생성하고 의존성을 설치했다.

## 실행 방법

### 정적 웹

```bash
source .venv/bin/activate
python serve.py
```

- `http://localhost:8000/`
- `http://localhost:8000/mgn-easy`

### Streamlit 통합 앱

```bash
source .venv/bin/activate
streamlit run app.py
```

## 참고

현재 `requirements.txt`는 기존 파일이 없어서 코드 import와 실제 설치된 패키지 버전을 기준으로 생성했다.
