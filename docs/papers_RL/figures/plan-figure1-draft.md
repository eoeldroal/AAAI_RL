# StreamWeave Figure 1 Draft Implementation Plan (2026-07-24 개정)

> **For agentic workers:** 이 plan의 구 버전(naive-composition double bind 구도)은 폐기되었다.
> 이 파일이 Figure 1의 저장소 내 canonical 의도와 기각 이력을 소유한다. 편집 가능한 DC HTML은
> 외부 디자인 세션에 있으며 저장소 경로가 아니다. 이 plan을 처음부터 다시 실행하지 말고,
> [`figures_README.md`](figures_README.md)의 상태와 함께 현재 디자인 위에서 작업한다.

**Goal:** 두 병목(compute·signal)에 대한 두 부분 처방과 StreamWeave의 자리를 한 그림으로 보인다.

## 확정 구도 (구 구도 대체)

세 행이 같은 all-✗ hard prompt group을 놓고 비교된다:

1. **Synchronous, expert-supervised** — barrier에서 group 완성 → all-fail → expert τ* 주입.
   신호 ✓ / barrier까지 trainer idle ✗.
2. **Fully-asynchronous RL** — back-to-back 생성·연속 학습으로 idle 없음 ✓ / all-fail group은
   reward 동일 → advantage 0, expert 채널 부재 → 이 group에서 학습 신호 없음 ✗.
3. **StreamWeave** — back-to-back 생성 유지, 완성 눈금(점선 boundary)에서 group 재구성 →
   분기(any ✓ → RL / all ✗ → τ*) → 신호 ✓, idle 없음 ✓.

## 기각 이력 (되돌아가지 말 것)

- "naive async = 조기 발화(2 of 4 arrived)" — 사실 오류. fully-async RL은 group을 조기 판정하지 않는다.
- "naive async = group 복원 실패" — 부정확. 실패의 정체는 복원이 아니라 학습 신호의 부재다.
- 좌측 premise 패널(double bottleneck 별도 다이어그램) — 본문 중복, 상단 이탤릭 한 줄로 대체.

## Global Constraints

- `HPT`, framework 이름, queue 내부, objective 수식, 측정 수치는 Figure 1에 넣지 않는다.
- 라벨은 영어, white flat vector-first academic 스타일.
- 색 규약은 mono+gold (`figures_README.md`의 2026-07-24 개정 규약을 따른다).
- Nonblocking 실행이 보이도록 back-to-back trajectory(익명 filler 포함)를 유지한다.
- Row C의 분기는 미니 group 카드 1개 + 화살표 2개까지만 — gate 박스·γ 수식·store는
  Figure 2의 몫이다.
- 해석 문장은 캡션으로 (사용자가 직접 작성).

## 남은 작업

- [ ] 분기 라벨 표기 확정 (`any ✓ / all ✗` 평이형 vs `P>γ / P≤γ` 수식형)
- [ ] 최종 크기(2단 폭) 글자 비율·흑백 인쇄 점검
- [ ] 확정 후 SVG/PDF/PNG export 및 `docs/papers_RL/figures/` 반입
