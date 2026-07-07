Project 4. AI Agent-based Soccer Shooting Analysis Prototype

단일 카메라로 촬영한 축구 슈팅 영상에서 공의 움직임을 추적하고, 카메라 보정 정보와 단순화된 물리 모델을 활용해 속도와 궤적을 추정·시각화하는 웹 서비스 프로토타입이다. 메인 포트폴리오가 아니라, AI Agent를 활용한 빠른 CV 서비스 프로토타이핑과 FastAPI 기반 AI 분석 백엔드 흐름을 보여주는 보조 프로젝트로 정리했다.

백엔드는 FastAPI 기반으로 영상 업로드와 분석 작업을 처리하고, 분석 진행률은 SSE(Server-Sent Events)를 통해 React 프론트엔드에 전달한다. 분석 로직은 YOLOv8 기반 공 검출, OpenCV CSRT tracking, YOLOv8-seg 기반 골대 영역 탐지, SciPy least_squares 기반 궤적 파라미터 피팅으로 구성했다. 결과는 추정 속도와 궤적 오버레이 이미지 형태로 사용자에게 제공된다.

AI Agent는 요구사항 분해, FastAPI/SSE 구조 초안 작성, CV 분석 파이프라인 설계 보조, React UI 구성, 오류 원인 후보 정리, README 문서화에 활용했다. 다만 단일 카메라 기반 분석의 Depth Ambiguity와 Ground Truth 부재를 고려해, 산출 결과를 정밀 측정값이 아니라 추정값으로 포지셔닝했다.

이 프로젝트를 통해 무거운 AI/CV 분석 작업을 웹 서비스 흐름에 연결할 때, 단순히 모델을 실행하는 것뿐 아니라 업로드 처리, 비동기 분석, 진행률 스트리밍, 결과 시각화, 한계 고지가 함께 필요하다는 점을 경험했다.
