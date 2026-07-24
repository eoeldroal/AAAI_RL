# StreamWeave Diagram Set Implementation Plan (2026-07-24 개정)

> **For agentic workers:** 이 plan의 Task 1·2가 만든 구 승인본
> (`figure1_streamweave_overview`, `figure2_training_pipeline`)은 폐기되었다.
> 이 문서와 [`figures_README.md`](figures_README.md)가 저장소 안의 canonical 의도 기록이다.
> 편집 가능한 DC HTML은 외부 디자인 세션에 있으며 저장소 경로가 아니다. 최종 export가 반입되기
> 전까지 그림 상태는 `DESIGN LOCKED / EXPORT PENDING`이다.

**Goal:** 하나의 시각 언어를 공유하는 Figure 1(문제 서사)과 Figure 2(end-to-end pipeline).

## 폐기 사유 요약

- 구 Figure 1: "Wait→idle / Decide early→wrong source" 프레임이 사실 오류
  (실패의 정체는 학습 신호의 부재).
- 구 Figure 2: "RL operators / SFT update 두 박스 + ⊕"가 코드 사실과 불일치 —
  `losses.py`의 loss는 branch-blind 단일 경로이고 per-branch weight는 구조적으로 거부된다.
  β·reference·mask는 conversion(assembler)이 row에 실어 보낸다. "SFT"는 내부 용어.

## Global Constraints (개정)

- Repository test suite를 실행하지 않는다.
- 그림 안 가시 텍스트는 최소 컴포넌트 라벨로 제한, 해석은 캡션.
- Framework queue 크기, Ray 이름, trim-and-carryover, tensor 필드는 넣지 않는다.
- **색 규약: mono+gold** (`figures_README.md` 2026-07-24 개정) — 구 blue/orange/teal/coral 규약 폐지.
- 물리 컨테이너는 5개 동급 단계(Generator / Reconstruction·source decision / Routed-group
  stream / Conversion / Trainer), StreamWeave 단 2개는 gold 밑줄로만 구별.
  실제 프로세스 토폴로지(경계 단 2개가 generator/trainer 프로세스 내부)는 캡션이 책임진다.
- Trainer는 mixed batch를 받아 **단일 policy-gradient update** 하나로 그린다. ⊕ 노드 금지.
- 상속 async 기전을 standalone novelty로 보이게 하지 않는다.

## 공유 시각 어휘 (두 그림 공통)

- 완성 눈금: 점선 세로선 = barrier 없는 boundary (sync의 실선 barrier와 대비)
- gold = expert supervision·학습 신호, neutral = policy, ✗ 글리프 = 실패
- Gantt bar 문법의 back-to-back trajectory 실행
- 회색 dashed = parameter refresh (control plane)

## 남은 작업

- [ ] 두 초안 확정 후 SVG/PDF/PNG export, `docs/papers_RL/figures/` 반입, README 표 갱신
- [ ] 두 PNG를 나란히 놓고 이 문서의 공유 시각 어휘와 constraints를 기준으로 최종 점검
- [ ] PDF Poppler 렌더에서 폰트 치환·crop 확인
- [ ] 모든 가시 명사가 `Codemap_RL.md`와 구현의 실제 단계에 대응하는지 확인
