import streamlit as st
import pandas as pd
from openai import OpenAI
import os
import time

# --- 1. CẤU HÌNH TRANG WEB ---
st.set_page_config(
    page_title="Tuan Finance AI",
    page_icon="📈",
    layout="centered"
)

# --- 2. LẤY API KEY TỪ RENDER ---
api_key = os.environ.get("OPENAI_API_KEY")

# --- 3. DATABASE KHÁCH HÀNG (GIẢ LẬP) ---
VALID_KEYS = {
    "VIP888": {"name": "Admin Tuấn", "quota": 999},
    "KH01":   {"name": "Khách mời", "quota": 5}
}

# --- 4. HÀM XỬ LÝ DỮ LIỆU THẬT (TỪ EXCEL) ---
def get_stock_data(ticker):
    try:
        # Đọc file Excel (File này phải nằm cùng thư mục với app.py)
        df = pd.read_excel('Price_Vol.xlsx')
        
        # Chuẩn hóa tên cột (xóa khoảng trắng thừa nếu có)
        df.columns = df.columns.str.strip()
        
        # Tìm dòng chứa mã chứng khoán
        stock_row = df[df['Ticker'] == ticker]
        
        if stock_row.empty:
            return None, "Không tìm thấy mã này trong file Excel."
        
        # Lấy dữ liệu ra
        price = float(stock_row.iloc[0]['Close'])
        support = float(stock_row.iloc[0]['Support'])
        resistance = float(stock_row.iloc[0]['Resistance'])
        
        # Tính toán R:R
        # Rủi ro (Risk) = Giá mua - Cắt lỗ (Hỗ trợ)
        risk = price - support
        # Lợi nhuận (Reward) = Chốt lời (Kháng cự) - Giá mua
        reward = resistance - price
        
        if risk <= 0: # Trường hợp giá đã thủng hỗ trợ
            rr_ratio = 0
            verdict = "QUAN SÁT (Giá thủng hỗ trợ)"
        else:
            rr_ratio = round(reward / risk, 2)
            
            # Ra quyết định đơn giản
            if rr_ratio >= 2.0:
                verdict = "MUA MẠNH (R:R Hấp dẫn)"
            elif rr_ratio >= 1.0:
                verdict = "MUA THĂM DÒ"
            else:
                verdict = "BỎ QUA (Rủi ro cao)"

        return {
            "price": price,
            "support": support,
            "resistance": resistance,
            "rr": rr_ratio,
            "verdict": verdict
        }, None

    except Exception as e:
        return None, f"Lỗi đọc dữ liệu: {str(e)}"

# --- 5. GIAO DIỆN CHÍNH (STREAMLIT) ---
st.title("📈 AI STOCK MASTER")
st.markdown("### Hệ thống phân tích & Định giá chuyên sâu (Real-time Data)")

# Khu vực nhập liệu
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        user_key = st.text_input("🔑 Nhập Mã VIP:", type="password")
    with col2:
        ticker = st.text_input("🔍 Mã cổ phiếu (VD: HPG):").upper()

# Nút bấm xử lý
if st.button("🚀 Phân Tích Ngay", type="primary"):
    
    # A. Kiểm tra quyền truy cập
    if user_key not in VALID_KEYS:
        st.error("❌ Mã VIP không đúng!")
    elif VALID_KEYS[user_key]['quota'] <= 0:
        st.error("⚠️ Tài khoản hết lượt.")
    elif not ticker:
        st.warning("Vui lòng nhập mã cổ phiếu.")
    else:
        # B. Bắt đầu chạy logic thật
        user_info = VALID_KEYS[user_key]
        
        # Thanh loading giả lập cho mượt
        progress_text = "Đang quét dữ liệu từ file Excel..."
        my_bar = st.progress(0, text=progress_text)
        for i in range(100):
            time.sleep(0.01)
            my_bar.progress(i + 1, text=progress_text)
            
        # Gọi hàm lấy dữ liệu thật
        data, error = get_stock_data(ticker)
        
        if error:
            st.error(f"❌ {error}") # Báo lỗi nếu không tìm thấy mã hoặc lỗi file
        else:
            # C. Gọi AI viết nhận định
            ai_comment = "Chưa kết nối AI."
            if api_key:
                try:
                    client = OpenAI(api_key=api_key)
                    prompt = f"""
                    Dữ liệu mã {ticker}: Giá {data['price']}, Hỗ trợ {data['support']}, Kháng cự {data['resistance']}, R:R {data['rr']}.
                    Khuyến nghị của hệ thống: {data['verdict']}.
                    Hãy viết 3 câu nhận định ngắn gọn, sắc sảo cho nhà đầu tư.
                    """
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": prompt}]
                    )
                    ai_comment = response.choices[0].message.content
                except Exception as e:
                    ai_comment = f"Lỗi kết nối AI: {str(e)}"

            # D. Hiển thị kết quả
            st.divider()
            st.success(f"✅ Kết quả phân tích mã {ticker}")
            
            # Hàng 1: Metric chính
            kpi1, kpi2, kpi3 = st.columns(3)
            kpi1.metric("Giá Đóng Cửa", f"{data['price']:,}")
            kpi2.metric("R:R Ratio", f"{data['rr']}x")
            
            # Tô màu khuyến nghị
            color = "off"
            if "MUA" in data['verdict']: color = "normal" 
            if "BỎ QUA" in data['verdict']: color = "inverse"
            kpi3.metric("Khuyến Nghị", data['verdict'], delta_color=color)
            
            # Hàng 2: Bảng chi tiết
            st.subheader("📊 Các mốc quan trọng")
            trade_df = pd.DataFrame({
                "Vùng Hỗ Trợ (Stoploss)": [f"{data['support']:,}"],
                "Giá Hiện Tại (Entry)": [f"{data['price']:,}"],
                "Vùng Kháng Cự (Target)": [f"{data['resistance']:,}"]
            })
            st.table(trade_df)
            
            # Hàng 3: AI Insight
            st.info(f"🤖 **Góc nhìn AI:** {ai_comment}")
            
            # Trừ quota
            VALID_KEYS[user_key]['quota'] -= 1