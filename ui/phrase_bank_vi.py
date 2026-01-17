from __future__ import annotations

"""Vietnamese phrase bank (client-facing) for keyed narrative."""

from typing import Dict, List


PHRASE_BANK_VI: Dict[str, List[str]] = {
    # DNA
    "DNA_BALANCED": [
        "{ticker} là nhóm cân bằng: thuận lợi khi đi đúng cấu trúc, nhưng hiệu quả vẫn phụ thuộc kỷ luật điểm rủi ro.",
    ],
    "DNA_DEFENSIVE": [
        "{ticker} thiên về phòng thủ: phù hợp ưu tiên bảo toàn và chọn điểm vào có biên an toàn.",
    ],
    "DNA_GLASS": [
        "{ticker} thuộc nhóm biến động mạnh: lợi nhuận có thể nhanh, đổi lại cần kiểm soát nhịp và quy mô chặt hơn.",
    ],
    "DNA_EVENT": [
        "{ticker} nhạy tin và rủi ro gap: nên ưu tiên kịch bản có mốc xác nhận rõ ràng và hạn chế cam kết sớm.",
    ],
    "DNA_ILLIQUID": [
        "{ticker} có đặc điểm thanh khoản/độ nhiễu cao: cần chấp nhận trượt giá và giảm kỳ vọng về độ mượt của điểm vào.",
    ],

    # Zone
    "ZONE_POSITIVE": [
        "Giá đang ở vùng thuận lợi của kế hoạch; xác suất đi theo hướng tích cực sẽ cao hơn nếu giữ được vùng này.",
    ],
    "ZONE_RECLAIM": [
        "Giá đang ở vùng cần tái chiếm; nhịp chủ động nên đến sau khi xác nhận vượt lại vùng reclaim.",
    ],
    "ZONE_RISK": [
        "Giá đang tiến gần/vào vùng rủi ro; trọng tâm chuyển sang bảo vệ và giảm sai số.",
    ],
    "ZONE_NEUTRAL": [
        "Giá ở vùng trung tính; chiến lược hợp lý là kiên nhẫn chờ điểm xác nhận tốt hơn.",
    ],

    # Bias
    "BIAS_AGGRESSIVE": [
        "Bias nghiêng chủ động, nhưng vẫn ưu tiên triển khai theo xác nhận của vùng giá thay vì đuổi theo cảm xúc.",
    ],
    "BIAS_CAUTIOUS": [
        "Bias nghiêng thận trọng: đi chậm để tối ưu xác suất và tránh tăng cam kết khi chưa đủ điều kiện.",
    ],
    "BIAS_DEFENSIVE": [
        "Bias nghiêng phòng thủ: ưu tiên quản trị rủi ro trước, hiệu quả sẽ đến sau khi thị trường ổn định hơn.",
    ],

    # Size
    "SIZE_FULL": [
        "Gợi ý quy mô: có thể dùng quy mô đầy đủ nếu các mốc rủi ro vẫn được giữ chặt.",
    ],
    "SIZE_PARTIAL": [
        "Gợi ý quy mô: ưu tiên tham gia một phần để giữ dư địa xử lý nếu thị trường đổi kịch bản.",
    ],
    "SIZE_PROBE": [
        "Gợi ý quy mô: thăm dò nhỏ để quan sát phản ứng tại vùng giá, trước khi nâng cam kết.",
    ],
    "SIZE_FLAT": [
        "Gợi ý quy mô: hạn chế hoặc đứng ngoài để tránh trả học phí không cần thiết.",
    ],

    # Decision (authority)
    "DECISION_BUY": [
        "Decision Layer đang nghiêng về MUA theo kế hoạch; điểm quan trọng là giữ kỷ luật tại mốc rủi ro.",
    ],
    "DECISION_HOLD": [
        "Decision Layer đang nghiêng về GIỮ; ưu tiên theo dõi vùng giá then chốt để quyết định có nâng cam kết hay không.",
    ],
    "DECISION_WAIT": [
        "Decision Layer đang nghiêng về CHỜ; xác nhận rõ ràng sẽ giúp tối ưu hoá tỷ lệ rủi ro/lợi nhuận.",
    ],
    "DECISION_TRIM": [
        "Decision Layer đang nghiêng về HẠ TỶ TRỌNG; mục tiêu là giảm rủi ro danh mục khi cấu trúc chưa đủ chắc.",
    ],
    "DECISION_EXIT": [
        "Decision Layer đang nghiêng về THOÁT; ưu tiên bảo toàn để chờ cơ hội có xác suất tốt hơn.",
    ],
    "DECISION_AVOID": [
        "Decision Layer đang nghiêng về TRÁNH; rủi ro hiện tại chưa xứng đáng với phần thưởng.",
    ],
    "DECISION_UNKNOWN": [
        "Decision Layer chưa đủ rõ ràng; nên ưu tiên quan sát và giữ kỷ luật rủi ro.",
    ],

    # Safety (must include uncertainty)
    "SAFETY_LINE": [
        "Thị trường luôn có độ bất định; phương án an toàn là bám mốc rủi ro và sẵn sàng đổi kịch bản nếu bị phá vỡ.",
        "Không có kịch bản nào chắc chắn; kỷ luật tại vùng rủi ro sẽ quyết định chất lượng kết quả.",
        "Xác suất không bao giờ là tuyệt đối; giữ kỷ luật rủi ro giúp bạn tồn tại để tận dụng nhịp thuận lợi.",
    ],
}
