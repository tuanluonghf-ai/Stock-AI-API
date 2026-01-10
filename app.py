from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import streamlit as st

APP_TITLE = os.environ.get("INCEPTION_APP_TITLE", "INCEPTION")
APP_VERSION = os.environ.get("INCEPTION_APP_VERSION", "16.0")

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def _require_path(p: Path, hint: str) -> None:
    if not p.exists():
        raise RuntimeError(f"Missing required path: {p}. {hint}")

_require_path(REPO_ROOT / "inception" / "__init__.py",
              "Ensure package inception/ exists at repo root and has __init__.py")

from inception.core.pipeline import build_result as build_result_pipeline
from inception.core.report_ad_builder import generate_insight_report
from inception.modules import load_default_modules
from inception.ui.renderers import render_appendix_e, render_report_pretty
from inception.ui.styles import GLOBAL_CSS, render_header_html

# Ensure module registry is loaded (required bootstrap for modular engine)
load_default_modules()

PRICE_VOL_PATH = "Price_Vol.xlsx"
BASE_DIR = REPO_ROOT
DATA_DIR = Path(os.environ.get("INCEPTION_DATA_DIR", str(BASE_DIR))).resolve()

def resolve_data_path(path: str) -> str:
    if not path:
        return path
    p = Path(path)
    if p.is_absolute() and p.exists():
        return str(p)
    cand = (DATA_DIR / p).resolve()
    if cand.exists():
        return str(cand)
    cand2 = (Path.cwd() / p).resolve()
    if cand2.exists():
        return str(cand2)
    return str(cand)

VALID_KEYS = {
    "VIP888": {"name": "Admin Tuấn", "quota": 999},
    "KH01": {"name": "Khách mời 01", "quota": 5},
    "KH02": {"name": "Khách mời 02", "quota": 5},
    "KH03": {"name": "Khách mời 03", "quota": 5},
    "KH04": {"name": "Khách mời 04", "quota": 5},
    "KH05": {"name": "Khách mời 05", "quota": 5},
}

def _safe_text(x: Any) -> str:
    try:
        return "" if x is None else str(x)
    except Exception:
        return ""

def _to_float(x: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return default

def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide", page_icon="🟣")

    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
    st.markdown(render_header_html(APP_TITLE), unsafe_allow_html=True)
    st.divider()

    with st.sidebar:
        user_key = st.text_input("Client Code", type="password", placeholder="Client Code")
        ticker_input = st.text_input("Mã Cổ Phiếu:", value="VCB", key="ticker_input").upper()

        prev_ticker = st.session_state.get("__last_ticker")
        if prev_ticker is None:
            st.session_state["__last_ticker"] = ticker_input
        elif _safe_text(prev_ticker).strip().upper() != ticker_input:
            st.session_state["position_mode_radio"] = "Mua mới (FLAT)"
            st.session_state["avg_cost_inp"] = 0.0
            st.session_state["position_size_pct_inp"] = 0.0
            st.session_state["risk_budget_pct_inp"] = 1.0
            st.session_state["holding_horizon_inp"] = "Swing"
            st.session_state["timeframe_inp"] = "D"
            st.session_state["__last_ticker"] = ticker_input

        pm_label = st.radio(
            "Tình trạng vị thế",
            ["Mua mới (FLAT)", "Đang nắm giữ (HOLDING)"],
            index=0,
            key="position_mode_radio",
        )
        position_mode = "HOLDING" if str(pm_label).startswith("Đang") else "FLAT"

        with st.expander("Thông tin vị thế (tuỳ chọn)", expanded=(position_mode == "HOLDING")):
            avg_cost = st.number_input("Giá vốn (avg cost)", min_value=0.0, value=0.0, step=0.1, key="avg_cost_inp")
            position_size_pct = st.number_input("Tỷ trọng đang nắm (% NAV)", min_value=0.0, max_value=100.0, value=0.0, step=0.5, key="position_size_pct_inp")
            risk_budget_pct = st.number_input("Ngân sách rủi ro (% NAV / trade)", min_value=0.0, max_value=5.0, value=1.0, step=0.1, key="risk_budget_pct_inp")
            holding_horizon = st.selectbox("Holding horizon", ["Swing", "Position"], index=0, key="holding_horizon_inp")
            timeframe = st.selectbox("Timeframe", ["D", "W"], index=0, key="timeframe_inp")

        run_btn = st.button("Phân tích", type="primary", use_container_width=True)
        output_mode = "Character"

    if run_btn:
        if user_key not in VALID_KEYS:
            st.error("❌ Client Code không đúng. Vui lòng nhập lại.")
            return

        with st.spinner(f"Đang xử lý phân tích {ticker_input}..."):
            try:
                avg_cost_f: Optional[float] = _to_float(avg_cost, None)
                if avg_cost_f is not None and avg_cost_f <= 0:
                    avg_cost_f = None

                if position_mode != "HOLDING":
                    avg_cost_f = None
                    position_size_pct = 0.0

                result: Dict[str, Any] = build_result_pipeline(
                    ticker=ticker_input,
                    data_dir=str(DATA_DIR),
                    price_vol_path=resolve_data_path(PRICE_VOL_PATH),
                    position_mode=position_mode,
                    avg_cost=avg_cost_f,
                    position_size_pct_nav=float(position_size_pct) if isinstance(position_size_pct, (int, float)) else 0.0,
                    risk_budget_pct_nav=float(risk_budget_pct) if isinstance(risk_budget_pct, (int, float)) else 1.0,
                    holding_horizon=holding_horizon,
                    timeframe=timeframe,
                    enabled_modules=["character", "report_ad"],
                )

                report = ""
                try:
                    report = (result or {}).get("Report") if isinstance(result, dict) else ""
                except Exception:
                    report = ""
                if not report:
                    report = generate_insight_report(result if isinstance(result, dict) else {})

                st.markdown("<hr>", unsafe_allow_html=True)

                analysis_pack = result.get("AnalysisPack", {}) if isinstance(result, dict) else {}
                if not isinstance(analysis_pack, dict):
                    analysis_pack = {}

                if output_mode == "Character":
                    render_appendix_e(result, report, analysis_pack)
                else:
                    render_report_pretty(report, analysis_pack)

            except Exception as e:
                st.error(f"⚠️ Lỗi xử lý: {e}")
                try:
                    import traceback
                    with st.expander("Chi tiết lỗi (traceback)", expanded=False):
                        st.code(traceback.format_exc())
                except Exception:
                    pass
                st.exception(e)

    st.divider()
    st.markdown(
        f"""
        <p style='text-align:center; color:#6B7280; font-size:13px;'>
        © 2026 INCEPTION Research Framework<br>
        Phiên bản {APP_VERSION}
        </p>
        """,
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()
