# Q-Stack Phase 1: Reproducible Single-Confluence Falsification & Risk Engine

본 프로젝트는 `dini_codex_bot`, `dini_claude_bot`, `dini_opgo` 3자 에이전트 위원회의 최종 합의(SYNTHESIS)를 구현한 **Phase 1 검증 연구소**입니다.

## 핵심 설계 원칙
1. **단일 사전고정 전략 (Confluence 1-Set)**:
   - `BTC 베타 제거 잔차 |z| >= 2.5` + `Failed Breakout (스윙 고/저점 스윕 후 복귀)` + `CVD(Taker Buy/Sell Volume Delta) 흡수 확인`
2. **규칙형 기각 필터 (Rule-based Abstention, 비용 $0)**:
   - 기대 수익폭이 왕복 거래비용의 3배 미만일 때 `ABSTAIN`
   - BTC 급등/급락 방향과 역행할 때 `ABSTAIN`
3. **W1(알파 엣지 5항목) × W2(리스크 강제 2항목 + 누수 방지) 병렬 독립 판정**:
   - **W1**: 표본 $\ge 300$건(미달 시 24개월 확장/증거부족 판정), 순수익/총비용 $\ge 1.5$, 후반 30% OOS Sharpe $\ge 1.0$(표본 $\ge 90$건 조건), MDD $\le 15\%$, 1,000회 랜덤 진입 순열/부트스트랩 검정 $p < 0.05$.
   - **W2**: `shift(1)` 미래 참조 누수 0 검증, 1회 손실 캡(0.5~1.5%) 강제 절삭 검증, 일일 누적 손실(-2.0%) 도달 시 신규 진입 100% 차단(Kill Switch) 검증.
