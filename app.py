import streamlit as st
import pandas as pd
from openai import OpenAI
import os
import time

# --- 1. CẤU HÌNH ---
st.set_page_config(page_title="Tuan Finance AI", page_icon="📈", layout="centered")
api_key = os.environ.get("OPENAI_API_KEY")

VALID_KEYS = {
    "VIP888": {"name": "Admin Tuấn", "quota": 999},
    "KH01":   {"name": "Khách mời", "quota": 5}
}

# --- 2. HÀM XỬ LÝ DỮ LIỆU (ĐÃ FIX CHO KHỚP FILE CỦA ANH) ---
def get_stock_data(ticker):
    try:
        # Đọc file
        df = pd.read_excel('Price_Vol.xlsx')
        
        # Chuẩn hóa tên cột (Xóa khoảng trắng thừa, ví dụ "VMA 20 " -> "VMA 20")
        df.columns = [c.strip() for c in df.columns]
        
        # Tìm dòng chứa mã chứng khoán
        stock_row = df[df['Ticker'] == ticker]
        
        if stock_row.empty:
            return None, "Không tìm thấy mã này trong file dữ liệu."
        
        # Lấy dữ liệu (Dựa trên cột trong ảnh anh gửi)
        row = stock_row.iloc[0]
        price = float(row['Close'])
        low = float(row['Low'])     # Dùng giá Thấp nhất làm Hỗ trợ
        high = float(row['High'])   # Dùng giá Cao nhất làm Kháng cự
        volume = float(row['Volume'])
        
        # Xử lý cột VMA 20 (Có thể tên là "VMA 20" hoặc "VMA20")
        vma20 = 0
        if 'VMA 20' in row: vma20 = float(row['VMA 20'])
        elif 'VMA20' in row: vma20 = float(row['VMA20'])
        
        # --- TỰ ĐỘNG TÍNH TOÁN R:R ---
        # Chiến thuật: Mua tại giá đóng cửa, Cắt lỗ nếu thủng đáy (Low)
        support = low
        risk = price - support
        
        if risk <= 0: 
            risk = price * 0.01 # Tránh lỗi chia cho 0 nếu giá đóng cửa đúng bằng giá thấp nhất
            
        # Target kỳ vọng (Giả định tỷ lệ R:R chuẩn là 1:2)
        target_profit = price + (risk * 2.0)
        rr_ratio = round((target_profit - price) / risk, 2)
        
        # Đánh giá Volume
        vol_signal = "Đột biến" if volume > vma20 else "Trung bình"
        
        # Ra quyết định
        if rr_ratio >= 2.0 and vol_signal == "Đột biến":
            verdict = "MUA MẠNH (Tiền vào + R:R tốt)"
        elif rr_ratio >= 1.5:
            verdict = "MUA TÍCH LŨY"
        else:
            verdict = "QUAN SÁT THÊM"

        return {
            "price": price,
            "support": support,
            "resistance": target_profit,
            "volume": volume,
            "vol_signal": vol_signal,
            "rr": rr_ratio,
            "verdict": verdict
        }, None

    except FileNotFoundError:
        return None, "Lỗi: Không tìm thấy file 'Price_Vol.xlsx' trên hệ thống."
    except Exception as e:
        return None, f"Lỗi xử lý dữ liệu: {str(e)}"

# --- 3. GIAO DIỆN STREAMLIT ---
st.title("📈 AI STOCK MASTER")
st.markdown("### Phân tích dòng tiền & R:R")

# Nhập liệu
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        user_key = st.text_input("🔑 Nhập Mã VIP:", type="password")
    with col2:
        ticker = st.text_input("🔍 Mã cổ phiếu (VD: HPG):").upper()

# Xử lý
if st.button("🚀 Phân Tích Ngay", type="primary"):
    if user_key not in VALID_KEYS:
        st.error("❌ Mã VIP sai.")
    elif not ticker:
        st.warning("Vui lòng nhập mã.")
    else:
        # Load dữ liệu
        data, error = get_stock_data(ticker)
        
        if error:
            st.error(f"❌ {error}")
        else:
            # Gọi AI
            ai_text = "Chưa kết nối AI."
            if api_key:
                try:
                    client = OpenAI(api_key=api_key)
                    prompt = f"""
                    Mã {ticker}: Giá {data['price']}, Vol {data['vol_signal']}, R:R {data['rr']}.
                    Khuyến nghị: {data['verdict']}.
                    Viết 3 câu nhận định ngắn gọn cho nhà đầu tư.
                    """
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": prompt}]
                    )
                    ai_text = response.choices[0].message.content
                except: pass

            # Hiển thị
            st.divider()
            st.success(f"✅ Báo cáo mã: {ticker}")
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Giá Đóng Cửa", f"{data['price']:,}")
            c2.metric("Tín Hiệu Vol", data['vol_signal'], 
                      delta="Tốt" if data['vol_signal']=="Đột biến" else "Thường")
            c3.metric("Khuyến Nghị", data['verdict'])
            
            st.table(pd.DataFrame({
                "Hỗ Trợ (Stoploss)": [f"{data['support']:,}"],
                "Mục Tiêu (Target)": [f"{data['resistance']:,}"],
                "Tỷ lệ R:R": [f"{data['rr']}x"]
            }))
            
            st.info(f"🤖 **AI Nhận định:** {ai_text}")