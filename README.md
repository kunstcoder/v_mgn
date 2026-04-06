# MeshGraphNet(MGN) Demo + 쉬운 설명 웹페이지

이 저장소는 **MeshGraphNet(MGN) 개념 설명**과 **간단한 시각화 데모**를 함께 제공합니다.

- 정적 웹 데모: 입력/학습/출력 샘플 JSON을 화면에서 확인
- 쉬운 설명 페이지: MGN을 직관적으로 이해할 수 있는 웹 문서 제공
- Streamlit 통합 뷰어: mesh + 노드/엣지 상태 + step별 수렴 과정 시각화

---

## 1) 프로젝트 구조

- `index.html`: 기존 정적 데모 메인 페이지
- `mgn_easy.html`: MGN 쉬운 설명 페이지(신규)
- `styles.css`: 정적 페이지 공통 스타일
- `app.js`: 정적 데모용 샘플 데이터 로딩/렌더링 로직
- `serve.py`: 정적 페이지 서비스용 Python 서버(신규)
- `app.py`: Streamlit 통합 뷰어 (mesh + loss + 테이블)
- `data/`: 정적 데모에서 사용하는 샘플 JSON
- `demo/data_gen.py`: 합성 MeshGraph 데이터 생성
- `demo/streamlit_app.py`: Streamlit 기반 기본 시각화 앱
- `demo/training_sim.py`: 학습 시뮬레이션 보조 코드

---

## 2) 빠른 시작

### A. 정적 웹페이지 서비스 (추천)

아래 명령으로 정적 데모 서버를 실행합니다.

```bash
python serve.py
```

실행 후 접속:

- 기존 데모: <http://localhost:8000/>
- MGN 쉬운 설명 페이지: <http://localhost:8000/mgn-easy>

### B. Streamlit 통합 뷰어 실행

```bash
pip install streamlit pandas matplotlib numpy
streamlit run app.py
```

---

## 3) `mgn-easy` 페이지에서 볼 수 있는 내용

`/mgn-easy` 페이지는 MGN을 처음 접하는 사람도 이해하기 쉽게 다음을 설명합니다.

1. **왜 MGN이 필요한가**
   - 전통 CAE, 순수 데이터 AI, MGN의 차이
2. **MGN 핵심 파이프라인**
   - Mesh → Graph 변환
   - Message Passing 반복
   - 다음 상태 예측 및 롤아웃
3. **PINN 관점과의 연결**
   - 물리 일관성, 데이터 효율성, 일반화라는 공통 방향
4. **비유 기반 설명**
   - 도시 교통 네트워크 비유로 노드/엣지 상호작용 직관화

> 참고: 설명의 관점은 PINN 소개 글에서 강조한 "물리를 반영한 AI" 철학을 MGN 문맥으로 재구성한 것입니다.

---

## 4) 기존 정적 데모(`index.html`) 사용법

각 섹션 버튼을 눌러 샘플 데이터를 불러오면 됩니다.

- **Input Data**: 노드/엣지/피처 구성 예시
- **Training Process**: 학습 로그/지표 예시
- **Output Data**: 예측 상태(Current)와 목표 상태(Target) 비교

출력 섹션에서는 다음이 자동 표시됩니다.

- 현재 예측 텐서
- 목표 상태 텐서(비교 기준)
- 노드별 압력/속도 차이(`delta`)

---

## 5) Streamlit 통합 뷰어(`app.py`) 주요 기능

`streamlit run app.py`로 실행되는 통합 뷰어는 다음 기능을 포함합니다.

- seed, mesh type(triangle/quad), total steps, learning rate, noise scale 조절
- 학습 step 슬라이더 + 다음 step 버튼 + step 초기화
- 노드 색상(`pred_pressure`), 엣지 색상(`|Δpressure|`) 동시 시각화
- step별 MSE 곡선
- 노드/엣지 상세 수치 테이블

교육/설명용 데모이므로 수치 자체는 합성 데이터 기반입니다.

---

## 6) 문제 해결 팁

- 포트 충돌 시:
  - `serve.py`의 `PORT` 값을 변경해 다시 실행
- 정적 파일이 안 보일 때:
  - 서버를 실행한 터미널의 현재 경로가 저장소 루트인지 확인
- Streamlit 실행 오류:
  - 의존성 재설치 후 재시도
  - `pip install --upgrade streamlit pandas matplotlib numpy`

---

## 7) 한 줄 요약

이 저장소는 **MGN 개념을 빠르게 이해하고**, **샘플 데이터 기반 시각화까지 바로 실행**할 수 있도록 구성된 학습용 데모입니다.
