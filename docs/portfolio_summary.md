# AI Agent-based Soccer Shooting Analysis Prototype

## Project Summary

단일 카메라로 촬영한 축구 슈팅 영상에서 공의 움직임을 추적하고, 카메라 보정 정보와 단순화된 물리 모델을 활용해 속도와 궤적을 추정·시각화하는 웹 서비스 프로토타입입니다.

## Implementation Scope

- FastAPI 기반 영상 업로드 및 분석 API
- SSE 기반 분석 진행률 스트리밍
- React/Vite 기반 업로드 및 결과 UI
- YOLOv8 기반 공 검출
- OpenCV CSRT tracking 보완
- YOLOv8-seg 기반 골대 영역 탐지
- SciPy least_squares 기반 궤적 파라미터 피팅
- 결과 오버레이 이미지 생성

## Portfolio Positioning

이 프로젝트는 메인 포트폴리오가 아니라, AI Agent를 활용한 빠른 CV 서비스 프로토타이핑과 FastAPI 기반 AI 분석 백엔드 흐름을 보여주는 보조 프로젝트입니다.
