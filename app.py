import streamlit as st
import pandas as pd
from openai import OpenAI
import time
import os

# --- CẤU HÌNH TRANG WEB ---
st.set_page_config(
    page_title="Tuan Finance AI",
    page_icon="📈",
    layout="centered"
)

# --- CẤU HÌNH API KEY (QUAN TRỌNG) ---
# Cách tốt nhất là cài trong Environment Variable của Render, 
# nhưng để test nhanh anh dán trực tiếp vào đây (nhớ bảo mật).
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
# --- DATABASE KHÁCH HÀNG GIẢ LẬP ---
VALID_KEYS = {
    "VIP888": {"name": "Admin Tuấn", "quota": 999},
    "KH01":   {"name": "Khách mời", "quota": 5}
}

# --- HÀM LOGIC (GIẢ LẬP - SAU NÀY GHÉP EXCEL CỦA ANH VÀO ĐÂY) ---
def analyze_stock_dummy(ticker):
    """
    Hàm này tạm thời trả về số liệu giả định để test giao diện.
    Sau khi Web chạy, ta sẽ ghép logic đọc file Price_Vol.xlsx vào sau.
    """
    # Logic giả: Giá random theo tên mã
    base_price = len(ticker) * 10000 
    return {
        "price": base_price + 500,
        "change": 1.2,
        "rr": 2.8,
        "stop_loss": base_price * 0.95,
        "take_profit": base_price * 1.15,
        "verdict": "MUA TÍCH LŨY"
    }

# --- GIAO DIỆN CHÍNH (STREAMLIT) ---
st.title("📈 AI STOCK MASTER")
st.markdown("### Hệ thống phân tích & Định giá chuyên sâu")

# 1. Khu vực nhập liệu
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        user_key = st.text_input("🔑 Nhập Mã VIP:", type="password")
    with col2:
        ticker = st.text_input("🔍 Mã cổ phiếu (VD: HPG):").upper()

# 2. Nút bấm xử lý
if st.button("🚀 Phân Tích Ngay", type="primary"):
    
    # Kiểm tra Key
    if user_key not in VALID_KEYS:
        st.error("❌ Mã VIP không đúng hoặc chưa kích hoạt.")
    elif VALID_KEYS[user_key]['quota'] <= 0:
        st.error("⚠️ Tài khoản hết lượt. Vui lòng gia hạn.")
    elif not ticker:
        st.warning("Vui lòng nhập mã cổ phiếu cần soi.")
    else:
        # Bắt đầu chạy
        user_info = VALID_KEYS[user_key]
        st.toast(f"Xin chào {user_info['name']}! Đang kết nối máy chủ...", icon="👋")
        
        # Thanh tiến trình giả lập cho chuyên nghiệp
        progress_text = "Đang tải dữ liệu thị trường..."
        my_bar = st.progress(0, text=progress_text)
        
        for percent_complete in range(100):
            time.sleep(0.01)
            my_bar.progress(percent_complete + 1, text=progress_text)
        
        # Lấy dữ liệu
        data = analyze_stock_dummy(ticker)
        
        # Gọi GPT viết nhận định
        try:
            client = OpenAI(api_key=OPENAI_API_KEY)
            prompt = f"""
            Tôi là chuyên gia tài chính. Dựa trên số liệu: 
            Mã {ticker}, Giá {data['price']}, R:R {data['rr']}, Khuyến nghị {data['verdict']}.
            Hãy viết một lời khuyên ngắn (3 câu), văn phong chuyên nghiệp, sắc sảo.
            """
            response = client.chat.completions.create(
                model="gpt-3.5-turbo", # Dùng 3.5 cho rẻ và nhanh, hoặc gpt-4o nếu muốn xịn
                messages=[{"role": "user", "content": prompt}]
            )
            ai_comment = response.choices[0].message.content
        except Exception as e:
            ai_comment = "Không thể kết nối AI lúc này. (Kiểm tra lại API Key)"

        # 3. HIỂN THỊ KẾT QUẢ
        st.divider()
        st.success(f"✅ Đã phân tích xong mã {ticker}")
        
        # Hàng 1: Chỉ số chính
        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric("Giá Hiện Tại", f"{data['price']:,} VNĐ", f"{data['change']}%")
        kpi2.metric("R:R Ratio", f"{data['rr']}x")
        kpi3.metric("Khuyến Nghị", data['verdict'], delta_color="normal")
        
        # Hàng 2: Kế hoạch giao dịch (Table)
        st.subheader("📋 Kế hoạch giao dịch")
        trade_plan = pd.DataFrame({
            "Vùng Mua": [f"{data['price']:,}"],
            "Cắt Lỗ (Stoploss)": [f"{data['stop_loss']:,}"],
            "Chốt Lời (Target)": [f"{data['take_profit']:,}"]
        })
        st.table(trade_plan)
        
        # Hàng 3: Góc nhìn AI
        st.info(f"🤖 **Góc nhìn AI:** {ai_comment}")
        
        # Trừ lượt dùng
        VALID_KEYS[user_key]['quota'] -= 1
        st.caption(f"Số lượt còn lại của bạn: {VALID_KEYS[user_key]['quota']}")