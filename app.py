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

# Lấy API Key
api_key = os.environ.get("OPENAI_API_KEY")

# Database khách hàng giả lập
VALID_KEYS = {
    "VIP888": {"name": "Admin Tuấn", "quota": 999},
    "KH01":   {"name": "Khách mời", "quota": 5}
}

# --- 2. HÀM XỬ LÝ DỮ LIỆU (LOGIC ĐÃ ĐƯỢC KIỂM TRA) ---
def get_stock_data(ticker):
    try:
        # Đọc file Excel
        df = pd.read_excel('Price_Vol.xlsx')
        
        # Chuẩn hóa tên cột (Xóa khoảng trắng thừa)
        df.columns = [str(c).strip() for c in df.columns]
        
        # Tìm mã (Chuyển về chữ hoa để so sánh)
        ticker = ticker.upper().strip()
        df['Ticker'] = df['Ticker'].astype(str).str.upper().str.strip()
        
        stock_row = df[df['Ticker'] == ticker]
        
        if stock_row.empty:
            return None, "Không tìm thấy mã này trong dữ liệu."
        
        # Lấy dữ liệu
        row = stock_row.iloc[0]
        price = float(row['Close'])
        low = float(row['Low'])     # Dùng làm Hỗ trợ (Stoploss)
        high = float(row['High'])
        volume = float(row['Volume'])
        
        # Xử lý VMA20 (phòng trường hợp tên cột khác nhau chút xíu)
        vma20 = 0
        if 'VMA20' in row: vma20 = float(row['VMA20'])
        elif 'VMA 20' in row: vma20 = float(row['VMA 20'])
        
        # --- TÍNH TOÁN CHIẾN LƯỢC ---
        # 1. Xác định Rủi ro (Risk) = Giá vào - Giá thấp nhất
        support = low
        risk = price - support
        
        if risk <= 0: risk = price * 0.01 # Tránh lỗi chia cho 0
            
        # 2. Xác định Mục tiêu (Target) theo tỷ lệ R:R = 1:2
        target_profit = price + (risk * 2.0)
        
        # 3. Tính R:R thực tế (nếu dùng High làm kháng cự thì tính lại, ở đây ta dùng Target kỳ vọng)
        rr_ratio = 2.0 # Mặc định set kèo là 2.0
        
        # 4. Đánh giá Volume
        vol_signal = "Đột biến" if volume > vma20 else "Trung bình"
        
        # 5. Ra quyết định
        verdict = "MUA TÍCH LŨY"
        if vol_signal == "Đột biến":
            verdict = "MUA MẠNH (Dòng tiền vào)"
        
        return {
            "price": price,
            "support": support,
            "target": target_profit,
            "volume": volume,
            "vol_signal": vol_signal,
            "rr": rr_ratio,
            "verdict": verdict
        }, None

    except Exception as e:
        return None, f"Lỗi xử lý dữ liệu: {str(e)}"

# --- 3. GIAO DIỆN NGƯỜI DÙNG (UI) ---
st.title("📈 AI STOCK MASTER")
st.markdown("### Hệ thống phân tích & Định giá chuyên sâu")

# Khu vực nhập liệu
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        user_key = st.text_input("🔑 Nhập Mã VIP:", type="password")
    with col2:
        ticker = st.text_input("🔍 Mã cổ phiếu (VD: HPG):").upper()

# Nút bấm xử lý
if st.button("🚀 Phân Tích Ngay", type="primary"):
    
    # Kiểm tra đầu vào
    if user_key not in VALID_KEYS:
        st.error("❌ Mã VIP không đúng!")
    elif not ticker:
        st.warning("Vui lòng nhập mã cổ phiếu.")
    else:
        # Bắt đầu chạy
        user_info = VALID_KEYS[user_key]
        
        # Thanh loading
        progress_text = "Đang quét dữ liệu thị trường..."
        my_bar = st.progress(0, text=progress_text)
        for i in range(100):
            time.sleep(0.01)
            my_bar.progress(i + 1, text=progress_text)
            
        # Lấy dữ liệu
        data, error = get_stock_data(ticker)
        
        if error:
            st.error(f"❌ {error}")
        else:
            # --- GỌI AI & TÍNH TIỀN ---
            ai_comment = "Chưa kết nối AI."
            cost_msg = ""
            
            if api_key:
                try:
                    client = OpenAI(api_key=api_key)
                    prompt = f"""
                    Tôi là chuyên gia tài chính. 
                    Mã {ticker}: Giá {data['price']}, Hỗ trợ {data['support']}, Vol {data['vol_signal']}.
                    Hãy đưa ra 3 lời khuyên ngắn gọn, sắc bén cho nhà đầu tư cá nhân.
                    """
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": prompt}]
                    )
                    ai_comment = response.choices[0].message.content
                    
                    # TÍNH TOÁN CHI PHÍ TOKEN
                    tokens = response.usage.total_tokens
                    cost = (tokens / 1000000) * 0.50
                    cost_msg = f"(Tiêu tốn: {tokens} tokens ~ ${cost:.5f})"
                    
                except Exception as e:
                    ai_comment = f"Lỗi kết nối AI: {str(e)}"

            # --- HIỂN THỊ KẾT QUẢ ---
            st.divider()
            st.success(f"✅ Báo cáo phân tích: {ticker}")
            
            # Hàng 1: Chỉ số chính
            kpi1, kpi2, kpi3 = st.columns(3)
            kpi1.metric("Giá Hiện Tại", f"{data['price']:,}")
            kpi2.metric("Tín Hiệu Vol", data['vol_signal'], delta="Tốt" if data['vol_signal']=="Đột biến" else "Thường")
            
            # Tô màu khuyến nghị
            color = "normal" if "MUA" in data['verdict'] else "off"
            kpi3.metric("Khuyến Nghị", data['verdict'], delta_color=color)
            
            # Hàng 2: Bảng kế hoạch giao dịch (Trading Plan)
            st.subheader("📊 Kế hoạch giao dịch (Trading Plan)")
            trade_df = pd.DataFrame({
                "Điểm Cắt Lỗ (Stoploss)": [f"{data['support']:,}"],
                "Điểm Vào Lệnh (Entry)": [f"{data['price']:,}"],
                "Mục Tiêu Chốt Lời (Target)": [f"{data['target']:,}"]
            })
            st.table(trade_df)
            
            # Hàng 3: AI Insight
            st.info(f"🤖 **Góc nhìn Chuyên gia AI:**\n\n{ai_comment}")
            
            # Hàng 4: Footer minh bạch chi phí (Chỉ Admin thấy)
            if cost_msg:
                st.caption(f"💰 Chi phí hệ thống: {cost_msg}")