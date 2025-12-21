import streamlit as st
import pandas as pd
from openai import OpenAI
import os
import time

# --- CẤU HÌNH ---
st.set_page_config(page_title="Debug Mode", page_icon="🛠️", layout="centered")
api_key = os.environ.get("OPENAI_API_KEY")

# --- HÀM XỬ LÝ ---
def get_data_debug(ticker):
    try:
        # 1. Đọc file
        df = pd.read_excel('Price_Vol.xlsx')
        df.columns = [str(c).strip() for c in df.columns] # Xóa khoảng trắng tên cột
        
        # DEBUG: Trả về 5 dòng đầu để xem
        preview = df.head()
        
        # 2. Tìm mã
        # Chuyển hết về chữ hoa để so sánh cho chuẩn
        df['Ticker'] = df['Ticker'].astype(str).str.upper().str.strip()
        ticker = ticker.upper().strip()
        
        stock_row = df[df['Ticker'] == ticker]
        
        if stock_row.empty:
            # Lấy danh sách 5 mã đầu tiên có trong file để gợi ý
            available = df['Ticker'].head(5).tolist()
            return None, f"Không tìm thấy mã '{ticker}'. Có phải ý bạn là: {available}?", preview
            
        # 3. Lấy dữ liệu
        row = stock_row.iloc[0]
        
        # In ra các cột tìm thấy để debug
        found_cols = row.index.tolist()
        
        # Lấy giá trị (Chấp nhận lỗi để hiện ra màn hình)
        price = float(row['Close'])
        low = float(row['Low'])
        high = float(row['High'])
        volume = float(row['Volume'])
        
        # Logic tính R:R
        support = low
        if price <= support: support = price * 0.95
        risk = price - support
        target = price + (risk * 2.0)
        rr = round((target - price) / risk, 2) if risk > 0 else 0
        
        return {
            "price": price,
            "support": support,
            "target": target,
            "rr": rr,
            "verdict": "MUA" if rr > 2 else "QUAN SÁT"
        }, None, preview

    except Exception as e:
        return None, f"LỖI CODE: {str(e)}", None

# --- GIAO DIỆN ---
st.title("🛠️ CHẾ ĐỘ KIỂM TRA LỖI")

# 1. Kiểm tra file Excel trước
st.subheader("1. Kiểm tra dữ liệu nguồn")
if st.button("📂 Đọc thử file Excel"):
    try:
        df_test = pd.read_excel('Price_Vol.xlsx')
        st.success("✅ Đã đọc được file Excel!")
        st.write("Dữ liệu 3 dòng đầu tiên:")
        st.dataframe(df_test.head(3))
        st.write("Tên các cột tìm thấy:", df_test.columns.tolist())
    except Exception as e:
        st.error(f"❌ Không đọc được file: {e}")

st.divider()

# 2. Kiểm tra Logic
st.subheader("2. Kiểm tra phân tích")
ticker = st.text_input("Nhập mã (VD: HPG):")

if st.button("🚀 Chạy phân tích"):
    if not ticker:
        st.warning("Chưa nhập mã.")
    else:
        data, error, preview = get_data_debug(ticker)
        
        if error:
            st.error(f"❌ {error}")
            if preview is not None:
                st.info("Dữ liệu thô đang có trong file:")
                st.dataframe(preview)
        else:
            st.success("✅ ĐÃ CHẠY THÀNH CÔNG!")
            st.json(data) # In kết quả dạng thô
            
            # Test AI
            if api_key:
                st.info("dang goi AI...")
                try:
                    client = OpenAI(api_key=api_key)
                    res = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role":"user", "content": f"Hello {ticker}"}]
                    )
                    st.write("AI Trả lời:", res.choices[0].message.content)
                except Exception as e:
                    st.error(f"Lỗi AI: {e}")