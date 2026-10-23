"""
Hermes Agent & Multi-Worker CLI Bridge (`qstack_cli.py`)
Allows Oracle Server Hermes Agents and Discord Worker Bots (`codex`, `claude`, `opgo`)
to control Q-Stack, retrieve structured role prompts, read Discord briefings, and trigger cycles.

Usage Examples (for Hermes Agent / Discord Bot):
  python qstack_cli.py cycle                # Run 1 full autonomous cycle & output Discord briefing
  python qstack_cli.py briefing             # Print latest Discord-ready Markdown briefing
  python qstack_cli.py worker-prompt --role critic   # Generate prompt for Critic Worker (Dr. Eggbot)
  python qstack_cli.py worker-prompt --role research # Generate prompt for Research Worker (Hatcha)
  python qstack_cli.py worker-prompt --role judge    # Generate prompt for Final Judge Worker
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

BASE_DIR = Path(__file__).parent
REPORTS_DIR = BASE_DIR / "reports"
BRIEFING_PATH = REPORTS_DIR / "latest_discord_briefing.md"
SUMMARY_PATH = REPORTS_DIR / "regime_comparison_summary.json"


WORKER_SYSTEM_PROMPTS = {
    "research": (
        "너는 Q-Stack의 [리서치 전용 워커 (Hatcha / Quant Researcher)]이다. "
        "너의 유일한 역할은 3축 장세 지표(Trend ADX/Slope, Volatility Rank, Breadth/Flow)와 "
        "10개 코인의 잔차 Z-Score 및 거래량 배수를 객관적인 수치로 요약하고, 현재 장세가 추세장인지 횡보장인지 보고하는 것이다. "
        "절대 희망회로를 섞지 말고 오직 데이터와 팩트만 브리핑하라."
    ),
    "critic": (
        "너는 Q-Stack의 [비판 전용 워커 (Dr. Eggbot / Devil's Advocate)]이다. "
        "너의 유일한 역할은 리서치나 찬성 측이 제안한 진입 시그널에 대해 '우리가 무엇을 틀렸을 수 있는가(What could we be wrong about?)'를 "
        "집요하게 파고드는 것이다. 왕복 거래비용(수수료+2티어 슬리피지+펀딩비), 역선택 위험, 거래량 부족, BTC 역행 위험을 근거로 "
        "기대수익이 비용의 2.5배를 넘지 못하면 가차 없이 기각(VETO) 의견을 내라."
    ),
    "judge": (
        "너는 Q-Stack의 [최종 심판 및 집행 워커 (Autonomous Portfolio Manager)]이다. "
        "리서치 워커의 장세 보고와 비판 워커(Dr. Eggbot)의 반론을 종합하여 24시간 코인 선물 시장에서 즉각적인 최종 결정을 내려라. "
        "단, 어떤 경우에도 W2 하드 리스크 엔진(1회 리스크 1.0% 상한, 일일 손실 -2.0% 킬스위치, 레버리지 5배 상한)을 위반하는 주문은 승인할 수 없다."
    ),
}


def cmd_cycle() -> int:
    res = subprocess.run([sys.executable, str(BASE_DIR / "run_full_cycle.py")], cwd=str(BASE_DIR))
    if res.returncode == 0 and BRIEFING_PATH.exists():
        print("\n" + BRIEFING_PATH.read_text(encoding="utf-8"))
    return res.returncode


def cmd_briefing() -> int:
    if not BRIEFING_PATH.exists():
        print("아직 생성된 브리핑이 없습니다. `python qstack_cli.py cycle`을 먼저 실행하세요.")
        return 1
    print(BRIEFING_PATH.read_text(encoding="utf-8"))
    return 0


def cmd_worker_prompt(role: str) -> int:
    sys_prompt = WORKER_SYSTEM_PROMPTS.get(role)
    if not sys_prompt:
        print(f"Unknown role: {role}")
        return 1
    briefing_text = BRIEFING_PATH.read_text(encoding="utf-8") if BRIEFING_PATH.exists() else "(브리핑 없음)"
    output = (
        f"=== [HERMES WORKER PROMPT : {role.upper()}] ===\n"
        f"[시스템 지침 (역할 고정)]\n{sys_prompt}\n\n"
        f"[현재 Q-Stack 실시간 상태 데이터]\n{briefing_text}\n"
        f"================================================\n"
    )
    print(output)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Q-Stack Hermes & Discord Worker CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("cycle", help="Run 1 full autonomous cycle and print Discord briefing")
    sub.add_parser("briefing", help="Print the latest Discord briefing markdown")

    wp = sub.add_parser("worker-prompt", help="Output role-locked prompt + live data for Hermes worker bots")
    wp.add_argument("--role", choices=["research", "critic", "judge"], required=True)

    args = parser.parse_args()
    if args.command == "cycle":
        sys.exit(cmd_cycle())
    elif args.command == "briefing":
        sys.exit(cmd_briefing())
    elif args.command == "worker-prompt":
        sys.exit(cmd_worker_prompt(args.role))


if __name__ == "__main__":
    main()
