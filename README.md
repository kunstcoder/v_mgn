# MeshGraphNet Demo

이 저장소는 **MeshGraphNet 입력/학습/출력 시각화**를 위한 최소 실행 가능한 데모 구조를 제공합니다.

## 구성

- `index.html`: 기존 정적 웹 데모 화면 구조
- `styles.css`: 정적 데모 레이아웃 및 UI 스타일
- `app.js`: 정적 데모용 샘플 데이터 렌더링 로직
- `data/`: 정적 데모용 샘플 JSON 데이터
- `demo/data_gen.py`: 합성 MeshGraph 데이터(노드/엣지/글로벌 상태 + 메시 연결) 생성
- `demo/streamlit_app.py`: Streamlit 테이블 + 2D 플롯 동시 시각화 앱

## Streamlit 데모 실행

```bash
pip install streamlit pandas matplotlib numpy
streamlit run demo/streamlit_app.py
```

### 제공 기능

1. 소규모 노드 집합(10~30개) 및 `edge_index` 자동 생성
2. 메시 셀 연결(삼각형/사각형)과 그래프 엣지 연결 동시 보관
3. 노드/엣지/글로벌 상태를 데이터클래스로 정의
4. `st.dataframe` 기반 초기 상태 테이블과 노드+엣지 2D 플롯을 한 화면에서 렌더링
