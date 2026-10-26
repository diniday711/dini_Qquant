"""
Hermes Agent & Operator Emergency Remote Control CLI (`qstack_cli.py`)
Allows the User, Oracle Server Hermes Agents, and Discord Worker Bots (`codex`, `claude`, `opgo`)
to control Q-Stack, inspect status, pause/resume new entries, emergency panic-close all positions,
retrieve structured role prompts, read Discord briefings, and trigger cycles.

Usage Examples:
  python qstack_cli.py status                      # Print current account, pause state, HWM floor, and open slots
  python qstack_cli.py pause                       # Immediately pause all new trade entries
  python qstack_cli.py resume                      # Resume autonomous trade entries
  python qstack_cli.py panic-close --live-testnet  # Emergency market-close all open positions & cancel open orders
  python qstack_cli.py cycle                       # Run 1 full autonomous cycle & output Discord briefing
  python qstack_cli.py briefing                    # Print latest Discord-ready Markdown briefing
  python qstack_cli.py worker-prompt --role critic # Generate prompt for Critic Worker (Dr. Eggbot)
"""
import argparse
from pathlib import Path
import subprocess
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

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


def cmd_status() -> int:
    from config import CONFIG
    from engines.binance_testnet_client import BinanceFuturesTestnetClient
    from engines.position_manager import PositionLifecycleManager, load_user_trading_rules

    rules = load_user_trading_rules()
    client = BinanceFuturesTestnetClient()
    bal = client.get_usdt_balance(CONFIG.initial_capital)
    pm = PositionLifecycleManager(str(BASE_DIR / "data" / "crumb_memory.db"))
    hwm = pm.update_and_check_equity_hwm(float(bal["available_balance_usdt"]))
    active = pm.get_active_positions()

    print("=== [Q-Stack v3.4 실시간 운영 상태 점검] ===")
    print(f"• 계좌 모드: {bal['mode']}")
    print(f"• 가용 잔고: ${bal['available_balance_usdt']:,.2f} USDT | 시드 구간: {hwm['tier_name']} (최대 {hwm['tier_max_slots']}슬롯)")
    print(f"• 최고 자산(HWM): ${hwm['peak_equity_usdt']:,.2f} USDT | 85% 보호 바닥선: ${hwm['locked_floor_usdt']:,.2f} USDT (잠금 여부: {hwm['account_floor_locked']})")
    print(f"• 매매 일시정지(Pause) 상태: {rules.get('trading_paused', False)} | 제외 코인: {rules.get('excluded_symbols', [])}")
    print(f"• 현재 보유 포지션 수: {len(active)}개")
    for p in active:
        print(
            f"  - [{p['symbol']}] {p['side']} | 진입가 ${p['entry_price']:,.4f} | 손절가 ${p['stop_price']:,.4f} | "
            f"1차익절 ${p['tp1_price']:,.4f} (TP1달성={p['tp1_hit']}) | 미실현손익 ${p['unrealized_pnl_usd']:+,.2f}"
        )
    return 0


def cmd_pause(paused: bool) -> int:
    from engines.position_manager import set_operator_Pause_state

    rules = set_operator_Pause_state(paused)
    state_kr = "⏸️ [일시정지(PAUSED) 활성화 - 모든 신규 진입 차단]" if paused else "▶️ [정상 가동(RESUMED) - 자율 매매 재개]"
    print(f"{state_kr} (user_trading_rules.json trading_paused={rules['trading_paused']})")
    return 0


def cmd_panic_close(live_testnet: bool) -> int:
    from engines.binance_testnet_client import BinanceFuturesTestnetClient
    from engines.position_manager import PositionLifecycleManager

    client = BinanceFuturesTestnetClient()
    pm = PositionLifecycleManager(str(BASE_DIR / "data" / "crumb_memory.db"))
    events = pm.panic_close_all_positions(testnet_client=client, execute_on_testnet=live_testnet)
    print("🚨 [비상 전량 청산 & 일시정지(PANIC CLOSE) 실행 완료]")
    if not events:
        print("  -> 현재 열려 있는 포지션이 없으며, 신규 진입 일시정지(trading_paused=True) 잠금이 설정되었습니다.")
    for ev in events:
        print(f"  -> {ev}")
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
    parser = argparse.ArgumentParser(description="Q-Stack Hermes & Operator Remote Control CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="Inspect current balance, HWM floor, pause state, and active positions")
    sub.add_parser("pause", help="Immediately pause all new trade entries")
    sub.add_parser("resume", help="Resume autonomous trade entries")
    pc = sub.add_parser("panic-close", help="Emergency market-close all open positions and pause trading")
    pc.add_argument("--live-testnet", action="store_true", help="Send live market close & cancel orders to Binance Testnet")

    sub.add_parser("cycle", help="Run 1 full autonomous cycle and print Discord briefing")
    sub.add_parser("briefing", help="Print the latest Discord briefing markdown")

    wp = sub.add_parser("worker-prompt", help="Output role-locked prompt + live data for Hermes worker bots")
    wp.add_argument("--role", choices=["research", "critic", "judge"], required=True)

    args = parser.parse_args()
    if args.command == "status":
        sys.exit(cmd_status())
    elif args.command == "pause":
        sys.exit(cmd_pause(True))
    elif args.command == "resume":
        sys.exit(cmd_pause(False))
    elif args.command == "panic-close":
        sys.exit(cmd_panic_close(args.live_testnet))
    elif args.command == "cycle":
        sys.exit(cmd_cycle())
    elif args.command == "briefing":
        sys.exit(cmd_briefing())
    elif args.command == "worker-prompt":
        sys.exit(cmd_worker_prompt(args.role))


if __name__ == "__main__":
    main()
