"""Deterministic phrase bank for narrative variety (no LLM).

This module reduces repetitive wording across batch outputs while keeping results
stable and reproducible.

Selection method:
- Stable hash (md5) of the provided seed string.
- Index = hash % len(phrases)

Constraints:
- No numeric literals in phrases.
- No price level hints.
- Professional, advisory tone.
"""

from __future__ import annotations

import hashlib
from typing import Dict, List


PHRASES: Dict[str, List[str]] = {
    # ----------------------------
    # Section A (Technical)
    # ----------------------------
    "A_CTX": [
        "Kịch bản {scen}; nền xu hướng (MA) {ma_reg}.",
        "Khung tham chiếu {scen}; nền MA ghi nhận {ma_reg}.",
        "Bối cảnh {scen}; xu hướng nền (MA) đang ở trạng thái {ma_reg}.",
    ],
    "A_ASSESS_1": [
        "RSI {rsi_state}; MACD {macd_state}; mức đồng thuận RSI+MACD {align}.",
        "Xung lực: RSI {rsi_state}; MACD {macd_state}; đồng thuận {align}.",
        "Đọc xung lực: RSI {rsi_state}; MACD {macd_state}; đồng thuận {align}.",
    ],
    "A_ASSESS_2": [
        "Fibonacci ngắn {fib_short} | dài {fib_long}; dòng tiền {vol_reg}.",
        "Fibonacci: ngắn {fib_short} / dài {fib_long}; trạng thái dòng tiền {vol_reg}.",
        "Khung Fib: ngắn {fib_short} – dài {fib_long}; dòng tiền {vol_reg}.",
    ],
    "A_QUANT_NOTE": [
        "Ghi chú định lượng: Điểm tổng hợp {ms_total} | Tin cậy {conv}.",
        "Tổng hợp nhanh: MasterScore {ms_total} | Conviction {conv}.",
        "Góc nhìn số liệu: MasterScore {ms_total} | Tin cậy {conv}.",
    ],
    "TECH_ACT_BULL": [
        "Theo xu hướng là ưu tiên; vào từng phần và chỉ gia tăng khi có xác nhận, tránh mua đuổi.",
        "Ưu tiên đi theo xu hướng; quản trị bằng kỷ luật, chỉ nâng tỷ trọng khi xác nhận rõ.",
        "Giữ thiên hướng theo xu hướng; tránh hưng phấn mua đuổi, tập trung vào kỷ luật xác nhận.",
    ],
    "TECH_ACT_BEAR": [
        "Thiên về phòng thủ; giảm quy mô và chờ cấu trúc ổn định trở lại trước khi hành động.",
        "Ưu tiên bảo toàn vốn; chỉ hành động khi có xác nhận đảo chiều đáng tin.",
        "Giữ tư thế phòng thủ; hạn chế tăng rủi ro cho tới khi điều kiện rõ ràng.",
    ],
    "TECH_ACT_NEUTRAL": [
        "Kiên nhẫn quan sát; chỉ hành động khi cấu trúc và xung lực đồng thuận.",
        "Giữ kỷ luật chờ xác nhận; tránh quyết định vội trong vùng nhiễu.",
        "Ưu tiên quan sát và phản ứng theo cấu trúc; không ép lệnh khi điều kiện chưa chín.",
    ],

    # ----------------------------
    # Section B (Fundamental)
    # ----------------------------
    "B_CTX_HAS": [
        "Dữ liệu cơ bản có sẵn trong gói hiện tại.",
        "Có thông tin cơ bản để làm neo tham chiếu.",
        "Có dữ liệu cơ bản để đối chiếu kỳ vọng.",
    ],
    "B_ASSESS": [
        "Khuyến nghị (tham khảo) {rec}; Giá mục tiêu (k) {target_k} | Upside {upside}.",
        "Tham khảo: {rec}; Mốc mục tiêu (k) {target_k} | Upside {upside}.",
        "Tóm tắt: {rec}; Giá mục tiêu (k) {target_k} | Upside {upside}.",
    ],
    "B_ACT_HAS": [
        "Dùng làm neo kỳ vọng trung hạn; vẫn ưu tiên kỷ luật kỹ thuật và quản trị rủi ro khi triển khai.",
        "Lấy làm tham chiếu kỳ vọng; triển khai vẫn theo kỷ luật kỹ thuật và quản trị rủi ro.",
        "Ghi nhận kỳ vọng trung hạn; quyết định triển khai vẫn dựa trên kỷ luật và quản trị rủi ro.",
    ],
    "B_CTX_NONE": [
        "Chưa có dữ liệu cơ bản trong gói hiện tại.",
        "Hiện chưa có thông tin cơ bản trong bộ dữ liệu.",
        "Chưa có dữ liệu cơ bản để đối chiếu kỳ vọng.",
    ],
    "B_ACT_NONE": [
        "Tạm thời dùng khung kỹ thuật và kỷ luật giao dịch; bổ sung dữ liệu cơ bản khi cần quyết định trung hạn.",
        "Ưu tiên khung kỹ thuật và quản trị rủi ro; bổ sung dữ liệu cơ bản khi chuyển sang quyết định trung hạn.",
        "Trước mắt vận hành theo kỷ luật kỹ thuật; khi cần tầm nhìn trung hạn, bổ sung dữ liệu cơ bản.",
    ],

    # ----------------------------
    # Section C (Trade Plan)
    # ----------------------------
    "C_CTX": [
        "Setup chính {setup}; trạng thái {plan_state}; RR {rr}.",
        "Khung kế hoạch: setup {setup}; trạng thái {plan_state}; RR {rr}.",
        "Điểm tựa kế hoạch: setup {setup}; trạng thái {plan_state}; RR {rr}.",
    ],
    "C_ASSESS_EXEC": [
        "Kỷ luật triển khai {act} ({urg}).",
        "Trạng thái vận hành {act} ({urg}).",
        "Ưu tiên vận hành {act} ({urg}).",
    ],
    "C_ASSESS_COMP_OK": [
        "Hoàn thiện kế hoạch {comp}.",
        "Mức độ hoàn thiện {comp}.",
        "Trạng thái hoàn thiện {comp}.",
    ],
    "C_ASSESS_COMP_MISS": [
        "Hoàn thiện kế hoạch {comp} (thiếu: {miss}).",
        "Mức độ hoàn thiện {comp} (thiếu: {miss}).",
        "Trạng thái hoàn thiện {comp} (thiếu: {miss}).",
    ],
    "C_NOTE": [
        "Ghi chú: {msg}.",
        "Lưu ý: {msg}.",
        "Nhắc nhanh: {msg}.",
    ],
    "C_TAGS": [
        "Tags: {tags}.",
        "Từ khóa: {tags}.",
        "Điểm nhấn: {tags}.",
    ],
    "PLAN_ACT_BUY": [
        "Giải ngân từng phần; chỉ gia tăng khi điều kiện vào lệnh được kích hoạt rõ ràng.",
        "Vào từng phần; chỉ nâng tỷ trọng khi điều kiện kích hoạt rõ ràng, tránh dồn lệnh.",
        "Triển khai từng phần; chỉ gia tăng khi có xác nhận, ưu tiên kiểm soát rủi ro.",
    ],
    "PLAN_ACT_TRIM": [
        "Hạ tỷ trọng để quản trị rủi ro; giữ phần lõi nếu cấu trúc còn giữ được.",
        "Giảm tỷ trọng để siết rủi ro; chỉ giữ phần lõi khi cấu trúc còn ổn.",
        "Ưu tiên hạ tỷ trọng; giữ phần lõi nếu cấu trúc vẫn còn hiệu lực.",
    ],
    "PLAN_ACT_EXIT": [
        "Ưu tiên thoát/giảm mạnh để bảo toàn vốn; chỉ cân nhắc lại khi cấu trúc phục hồi.",
        "Bảo toàn vốn là ưu tiên; chỉ cân nhắc lại khi cấu trúc phục hồi rõ ràng.",
        "Ưu tiên giảm mạnh/thoát; quay lại khi cấu trúc phục hồi và điều kiện rõ ràng.",
    ],
    "PLAN_ACT_HOLD": [
        "Giữ vị thế và theo dõi kỷ luật; chỉ hành động khi điểm xác nhận xuất hiện.",
        "Giữ vị thế; theo dõi kỷ luật và chỉ hành động khi có xác nhận rõ ràng.",
        "Duy trì vị thế; chỉ điều chỉnh khi điều kiện xác nhận xuất hiện.",
    ],
    "PLAN_ACT_WAIT": [
        "Kiên nhẫn quan sát; tránh hành động khi điều kiện chưa rõ.",
        "Ưu tiên quan sát; không ép lệnh khi điều kiện chưa đủ.",
        "Đứng ngoài quan sát; chỉ hành động khi điều kiện rõ ràng.",
    ],
}


def _stable_index(seed: str, n: int) -> int:
    if n <= 0:
        return 0
    s = (seed or "").encode("utf-8", errors="ignore")
    h = hashlib.md5(s).hexdigest()
    return int(h[:8], 16) % n


def pick_phrase(key: str, seed: str) -> str:
    """Pick one phrase deterministically for a given key and seed."""
    arr = PHRASES.get(key) or []
    if not arr:
        return ""
    i = _stable_index(f"{key}|{seed}", len(arr))
    return arr[i]
