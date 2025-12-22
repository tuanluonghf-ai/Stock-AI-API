import streamlit as st
import pandas as pd
from openai import OpenAI
import os
import time

# --- 1. CẤU HÌNH ---
st.set_page_config(
    page_title="Tuan Finance AI - Pro",
    page_icon="💎",
    layout="wide" # Chế độ màn hình rộng để đọc báo cáo cho sướng
)

api_key = os.environ.get("OPENAI_API_KEY")

VALID_KEYS = {
    "VIP888": {"name": "Admin Tuấn", "quota": 999},
    "KH01":   {"name": "Khách mời", "quota": 5}
}

# --- 2. HÀM XỬ LÝ SỐ LIỆU ---
def get_stock_data(ticker):
    try:
        df = pd.read_excel('Price_Vol.xlsx')
        df.columns = [str(c).strip() for c in df.columns]
        
        ticker = ticker.upper().strip()
        df['Ticker'] = df['Ticker'].astype(str).str.upper().str.strip()
        
        stock_row = df[df['Ticker'] == ticker]
        if stock_row.empty: return None, "Không tìm thấy mã này."
        
        row = stock_row.iloc[0]
        price = float(row['Close'])
        low = float(row['Low'])
        high = float(row['High'])
        volume = float(row['Volume'])
        
        # Xử lý VMA20
        vma20 = 0
        if 'VMA20' in row: vma20 = float(row['VMA20'])
        elif 'VMA 20' in row: vma20 = float(row['VMA 20'])
        
        # Tính toán tham số kỹ thuật
        support = low # Hỗ trợ cứng
        resistance = high # Kháng cự tạm thời
        risk = price - support
        if risk <= 0: risk = price * 0.01
        
        target = price + (risk * 2.0)
        rr_ratio = 2.0
        
        vol_assessment = "ĐỘT BIẾN (Tiền vào mạnh)" if volume > vma20 else "TRUNG BÌNH (Thanh khoản thấp)"
        trend = "TĂNG NGẮN HẠN" if price > low else "GIẰNG CO/GIẢM"

        return {
            "ticker": ticker,
            "price": price,
            "support": support,
            "resistance": resistance,
            "target": target,
            "volume": volume,
            "vma20": vma20,
            "vol_signal": vol_assessment,
            "trend": trend
        }, None

    except Exception as e:
        return None, f"Lỗi: {str(e)}"

# --- 3. GIAO DIỆN ---
st.title("💎 HỆ THỐNG PHÂN TÍCH CHỨNG KHOÁN CHUYÊN SÂU")
st.markdown("---")

# Sidebar bên trái để nhập liệu cho gọn
with st.sidebar:
    st.header("⚙️ Bảng Điều Khiển")
    user_key = st.text_input("Mã VIP:", type="password")
    ticker = st.text_input("Mã Cổ Phiếu:", value="HPG").upper()
    btn_run = st.button("🚀 PHÂN TÍCH CHUYÊN SÂU", type="primary")
    
    st.info("ℹ️ Hệ thống sử dụng dữ liệu Real-time từ Excel kết hợp AI Lập luận.")

# Màn hình chính
if btn_run:
    if user_key not in VALID_KEYS:
        st.error("❌ Sai mã VIP!")
    else:
        # Load dữ liệu
        with st.spinner('Đang đọc dữ liệu thị trường & Tính toán chỉ số...'):
            data, error = get_stock_data(ticker)
            time.sleep(1) # Giả lập delay chút cho chuyên nghiệp

        if error:
            st.error(error)
        else:
            # --- PHẦN QUAN TRỌNG NHẤT: PROMPT NÂNG CAO ---
            if api_key:
                try:
                    client = OpenAI(api_key=api_key)
                    
                    # Đây là "Kịch bản" ra lệnh cho AI viết dài
                    prompt = f"""
                    Bạn là một Chuyên gia phân tích tài chính cấp cao (CFA Charterholder) với 20 năm kinh nghiệm.
                    Hãy viết một bản báo cáo chi tiết dựa trên dữ liệu thật sau đây:
                    
                    DỮ LIỆU ĐẦU VÀO CỦA MÃ {data['ticker']}:
                    - Giá đóng cửa: {data['price']}
                    - Vùng Hỗ trợ gần nhất (Stoploss): {data['support']}
                    - Kháng cự / Mục tiêu kỳ vọng: {data['target']}
                    - Khối lượng (Volume): {data['volume']} (Đánh giá: {data['vol_signal']})
                    - Xu hướng giá hiện tại: {data['trend']}

                    YÊU CẦU ĐỊNH DẠNG BÁO CÁO (Bắt buộc dùng Markdown, trình bày chuyên nghiệp như Bloomberg):
                    
                    # 1. TỔNG QUAN TÍN HIỆU (SNAPSHOT)
                    - Tóm tắt nhanh tình trạng mã này trong 2 dòng.
                    - Đánh giá sức mạnh dòng tiền dựa trên Volume.

                    # 2. PHÂN TÍCH KỸ THUẬT & HÀNH ĐỘNG GIÁ (PRICE ACTION)
                    - Phân tích vị thế giá hiện tại so với hỗ trợ {data['support']}.
                    - Phân tích tâm lý thị trường (Bullish hay Bearish) dựa trên việc giá đang {data['trend']}.
                    - (Tự lập luận thêm về rủi ro nếu thủng hỗ trợ).

                    # 3. KỊCH BẢN GIAO DỊCH (TRADE PLAN) - Quan trọng nhất
                    Lập bảng kế hoạch chi tiết:
                    - **Vùng Mua (Buy Zone):** Quanh vùng {data['price']}
                    - **Cắt lỗ (Stoploss):** Tuyệt đối tuân thủ tại {data['support']}
                    - **Chốt lời (Take Profit):** Kỳ vọng tại {data['target']} (R:R = 1:2)
                    
                    # 4. KHUYẾN NGHỊ CUỐI CÙNG
                    - Đưa ra lời khuyên dứt khoát: MUA NGAY / CHỜ MUA / hay BÁN.
                    - Một câu châm ngôn đầu tư phù hợp với bối cảnh này.
                    """

                    with st.spinner('AI đang viết báo cáo chi tiết...'):
                        response = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.7 # Tăng sự sáng tạo lên một chút
                        )
                        report_content = response.choices[0].message.content
                        
                        # Tính tiền
                        tokens = response.usage.total_tokens
                        cost = (tokens / 1000000) * 0.50
                        
                    # HIỂN THỊ BÁO CÁO
                    st.success("✅ PHÂN TÍCH HOÀN TẤT")
                    
                    # Chia cột hiển thị số liệu thô trước
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Mã CK", data['ticker'])
                    c2.metric("Giá", f"{data['price']:,}")
                    c3.metric("Hỗ Trợ", f"{data['support']:,}")
                    c4.metric("Volume", data['vol_signal'])
                    
                    st.divider()
                    
                    # Hiển thị bài văn của AI
                    st.markdown(report_content)
                    
                    st.divider()
                    st.caption(f"📊 Report generated by OpenAI GPT-3.5 | Cost: ~${cost:.5f}")

                except Exception as e:
                    st.error(f"Lỗi kết nối AI: {e}")
            else:
                st.warning("Chưa nhập API Key!")