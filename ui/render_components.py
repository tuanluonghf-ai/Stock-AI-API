from __future__ import annotations
import streamlit as st

def sec_title(text: str):
    st.markdown(f'<div class="sec-title">{text}</div>', unsafe_allow_html=True)

def major_sec(text: str):
    st.markdown(f'<div class="major-sec">{text}</div>', unsafe_allow_html=True)

def muted(text: str):
    st.markdown(f'<div class="muted">{text}</div>', unsafe_allow_html=True)

def info_box(text: str):
    st.markdown(
        f'<div style="border:1px dashed #334155;border-radius:12px;padding:12px;color:#94A3B8;">{text}</div>',
        unsafe_allow_html=True
    )

def divider():
    st.markdown('<hr style="border:none;border-top:1px solid #1f2933;margin:12px 0;">', unsafe_allow_html=True)
