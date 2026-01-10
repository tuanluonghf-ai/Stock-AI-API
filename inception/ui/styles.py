"""
UI shell styles for INCEPTION Streamlit app.

This module is UI-only. Do NOT place any computation or core logic here.
"""

from __future__ import annotations


GLOBAL_CSS = r"""<style>
:root{
  --bg: #0B1426;
  --panel: #081A33;
  --panel2: #0F2A44;
  --fg: rgba(255,255,255,0.92);
  --muted: rgba(255,255,255,0.74);
  --border: rgba(255,255,255,0.14);
}

/* Global canvas */
html, body, .stApp,
[data-testid="stAppViewContainer"],
.main, .block-container {
  background: var(--bg) !important;
  color: var(--fg) !important;
  font-family: "Segoe UI", sans-serif;
}

/* Sidebar */
[data-testid="stSidebar"]{
  background: var(--bg) !important;
  border-right: 1px solid rgba(255,255,255,0.10);
}

/* Sidebar text (avoid wildcard that breaks input text contrast) */
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div{
  color: var(--fg) !important;
}

/* Sidebar inputs — enforce dark fields + readable text */
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] textarea{
  background: #0B1220 !important;
  color: #FFFFFF !important;
  border: 1px solid rgba(255,255,255,0.16) !important;
  border-radius: 10px !important;
}
[data-testid="stSidebar"] input::placeholder,
[data-testid="stSidebar"] textarea::placeholder{
  color: rgba(255,255,255,0.55) !important;
}

/* BaseWeb select (Streamlit selectbox/multiselect) */
[data-testid="stSidebar"] div[data-baseweb="select"] > div{
  background: #0B1220 !important;
  color: #FFFFFF !important;
  border-color: rgba(255,255,255,0.16) !important;
  border-radius: 10px !important;
}
[data-testid="stSidebar"] div[data-baseweb="select"] span{
  color: #FFFFFF !important;
}
/* Links */
a, a:visited{color: inherit !important; text-decoration:none !important;}
a:hover{text-decoration:none !important;}

/* Markdown baseline */
.stMarkdown p, .stMarkdown li, .stMarkdown span, .stMarkdown div{
  font-size: 17px !important;
  line-height: 1.55 !important;
  color: var(--fg) !important;
}
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3{
  color: #FFFFFF !important;
}

/* Header */
.incept-wrap{margin: 0 0 10px 0;}
.incept-header{
  display:flex;
  justify-content:space-between;
  align-items:center;
  padding: 10px 14px;
  background: var(--bg);
  border-bottom: 1px solid rgba(255,255,255,0.10);
}
.incept-brand{
  font-weight: 900;
  font-size: 34px;
  letter-spacing: 0.8px;
  color: #FFFFFF;
}
.incept-nav{display:flex;gap:18px;align-items:center;}
.incept-nav a{
  color:#FFFFFF !important;
  text-decoration:none !important;
  font-weight: 900;
  letter-spacing: 0.8px;
  font-size: 14px;
  opacity: 0.92;
}
.incept-nav a:hover{opacity:1.0; text-decoration:none !important;}

/* Primary action button (glossy black) */
.stButton>button{
  width:100%;
  background: linear-gradient(180deg, #111827 0%, #000000 100%);
  color:#FFFFFF !important;
  font-weight: 800;
  border-radius: 10px;
  height: 44px;
  border: 1px solid rgba(255,255,255,0.12);
  box-shadow: 0 6px 14px rgba(0,0,0,0.45);
}
.stButton>button:hover{
  background: linear-gradient(180deg, #0B1220 0%, #000000 100%);
  border: 1px solid rgba(255,255,255,0.18);
}

/* =========================
   GAME CHARACTER CARD (light card on dark canvas)
   ========================= */
.gc-card{border:1px solid #E5E7EB;border-radius:16px;padding:14px 14px 10px;background:#ffffff;}
.gc-head{display:block;margin-bottom:10px;}
.gc-title{font-weight:800;letter-spacing:.6px;font-size:12px;color:#6B7280;}
.gc-class{font-weight:800;font-size:18px;color:#111827;}
.gc-h1{font-weight:900;font-size:32px;color:#0F172A;line-height:1.2;}
.gc-blurb{margin-top:8px;font-size:18px;line-height:1.6;color:#334155;}
.gc-bline{margin-top:6px;font-size:18px;line-height:1.5;color:#334155;}
.gc-bline b{color:#0F172A;}
.gc-radar-wrap{display:flex;gap:14px;align-items:center;}
.gc-radar-svg{width:220px;height:220px;flex:0 0 auto;}
.gc-radar-metrics{flex:1;min-width:220px;}
.gc-radar-item{display:flex;justify-content:space-between;gap:10px;margin:4px 0;font-size:16px;color:#334155;}
.gc-radar-lab{font-weight:700;color:#334155;}
.gc-radar-val{font-weight:800;color:#0F172A;}
.gc-sec{margin-top:10px;padding-top:10px;border-top:1px dashed #E5E7EB;}
.gc-sec-t{font-weight:900;font-size:20px;color:#374151;margin-bottom:10px;}
.gc-row{display:flex;gap:10px;align-items:center;margin:6px 0;}
.gc-k{width:190px;font-size:20px;color:#374151;}
.gc-bar{flex:1;height:16px;background:#F3F4F6;border-radius:99px;overflow:hidden;}
.gc-fill{height:16px;background:linear-gradient(90deg,#2563EB 0%,#7C3AED 100%);border-radius:99px;}
.gc-v{width:96px;text-align:right;font-size:20px;color:#111827;font-weight:800;}
.gc-flag{display:flex;gap:8px;align-items:center;margin:6px 0;padding:6px 8px;background:#F9FAFB;border-radius:10px;border:1px solid #EEF2F7;}
.gc-sev{font-size:14px;font-weight:800;color:#111827;background:#E5E7EB;border-radius:8px;padding:2px 6px;}
.gc-code{font-size:14px;font-weight:800;color:#374151;}
.gc-note{font-size:17px;color:#6B7280;}
.gc-tags{display:flex;flex-wrap:wrap;gap:6px;margin-top:6px;}
.gc-tag{font-size:15px;background:#111827;color:#fff;border-radius:999px;padding:4px 10px;}
.gc-conv{display:grid;gap:6px;}
.gc-conv-tier,.gc-conv-pts{font-size:24px;color:#111827;font-weight:600;}
.gc-conv-guide{font-size:20px;color:#6B7280;line-height:1.35;}

/* =========================
   EXECUTIVE SNAPSHOT (DASHBOARD) — DARK THEME
   ========================= */
.es-card{
  border:1px solid rgba(255,255,255,0.16);
  border-radius:18px;
  padding:14px;
  background:#081A33;
  margin-top:8px;
  width:100%;
  box-sizing:border-box;
  box-shadow: 0 2px 10px rgba(0,0,0,0.20);
}
.es-head{display:flex;justify-content:space-between;gap:12px;align-items:flex-start;flex-wrap:wrap;}
.es-left{display:flex;flex-direction:column;gap:4px;}
.es-tline{display:flex;align-items:center;gap:10px;flex-wrap:wrap;}
.es-ticker{font-weight:900;font-size:26px;color:#FFFFFF;letter-spacing:0.4px;}
.es-price{font-weight:900;font-size:26px;color:#FFFFFF;}
.es-chg{
  font-weight:900;font-size:14px;padding:4px 10px;border-radius:999px;
  background:rgba(255,255,255,0.10);
  border:1px solid rgba(255,255,255,0.16);
  color:#FFFFFF;
}
.es-sub{font-size:16px;color:rgba(255,255,255,0.82);font-weight:800;}
.es-right{text-align:right;display:flex;flex-direction:column;gap:6px;}
.es-badge{
  font-weight:900;font-size:13px;padding:6px 10px;border-radius:999px;
  background:#0B1426;color:#FFFFFF;display:inline-block;
  border:1px solid rgba(255,255,255,0.14);
}
.es-kelly{
  font-weight:900;font-size:13px;padding:6px 10px;border-radius:999px;
  background:#0F2A44;color:#FFFFFF;border:1px solid rgba(255,255,255,0.18);
  display:inline-block;
}
.es-meta{font-size:13px;color:rgba(255,255,255,0.70);font-weight:800;}
.es-body{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-top:14px;}
@media(max-width: 980px){.es-body{grid-template-columns:1fr;}.es-right{text-align:left;}}
.es-panel{
  border:1px solid rgba(255,255,255,0.14);
  border-radius:16px;
  padding:12px;
  background:#0F2A44;
}
.es-pt{font-weight:950;font-size:14px;color:rgba(255,255,255,0.70);letter-spacing:0.7px;margin-bottom:8px;}
.es-metric{display:flex;justify-content:space-between;gap:10px;font-size:16px;margin:6px 0;}
.es-metric .k{color:rgba(255,255,255,0.78);font-weight:850;}
.es-metric .v{color:#FFFFFF;font-weight:950;}
.es-mini{height:10px;background:rgba(255,255,255,0.12);border-radius:99px;overflow:hidden;margin-top:6px;}
.es-mini>div{height:10px;background:linear-gradient(90deg,#2563EB 0%,#7C3AED 100%);border-radius:99px;}
.es-bline-wrap{margin-top:6px;}
.es-bline{font-size:13px;color:rgba(255,255,255,0.82);line-height:1.35;margin:2px 0;}
.es-sig-wrap{display:flex;justify-content:center;align-items:center;margin-top:10px;}
.es-sig-radar{flex:0 0 220px;}
.es-radar-svg{width:220px;height:220px;display:block;}
.es-sig-metrics{flex:1;}
.es-sig-row{display:flex;justify-content:space-between;gap:10px;font-size:14px;margin:4px 0;}
.es-sig-row .k{color:rgba(255,255,255,0.78);font-weight:850;}
.es-sig-row .v{color:#FFFFFF;font-weight:950;}
.es-note{font-size:14px;color:rgba(255,255,255,0.82);line-height:1.45;}
.es-bul{margin:6px 0 0 16px;padding:0;}
.es-bul li{margin:2px 0;font-size:14px;color:rgba(255,255,255,0.86);font-weight:650;}
.es-dot{width:10px;height:10px;border-radius:50%;display:inline-block;margin-right:8px;}
.es-dot.g{background:#22C55E;}
.es-dot.y{background:#F59E0B;}
.es-dot.r{background:#EF4444;}

/* Expander summary look like an action button */
div[data-testid="stExpander"] > details{border:0 !important; background:transparent !important;}
div[data-testid="stExpander"] > details > summary{
  background: linear-gradient(180deg, #111827 0%, #000000 100%);
  border: 1px solid rgba(255,255,255,0.14);
  border-radius: 12px;
  padding: 10px 14px;
}
div[data-testid="stExpander"] > details > summary:hover{opacity:0.98;}

/* MAJOR SECTION HEADERS */
.major-sec{
  background:#0b1426;
  border:3px solid #FFFFFF;
  border-radius:18px;
  padding:15px 20px;
  margin:30px 0px 20px 0px;
  color:#FFFFFF;
  font-weight:900;
  font-size:26px;
  letter-spacing:0.8px;
  text-transform:uppercase;
}
</style>"""


HEADER_HTML_TEMPLATE = r"""<div class="incept-wrap">
  <div class="incept-header">
    <div class="incept-brand">{APP_TITLE}</div>
    <div class="incept-nav">
      <a href="javascript:void(0)">CỔ PHIẾU</a>
      <a href="javascript:void(0)">DANH MỤC</a>
    </div>
  </div>
</div>"""


def render_header_html(app_title: str) -> str:
    return HEADER_HTML_TEMPLATE.replace("{APP_TITLE}", app_title)
