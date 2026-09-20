import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.chunking import ChunkingStrategyComparator, compute_similarity
from src.embeddings import _mock_embed

print("=== 1. BASELINE COMPARISON ===")
comp = ChunkingStrategyComparator()
files = ["buyer-return-refund-period.md", "buyer-return-conditions.md", "seller-return-response-sla.md"]
for fname in files:
    text = Path("data/ecommerce", fname).read_text(encoding="utf-8")
    body = text.split("---", 2)[2].strip()
    res = comp.compare(body, chunk_size=200)
    print(f"\nTài liệu: {fname}")
    for strat in ["fixed_size", "by_sentences", "recursive"]:
        cnt = res[strat]["count"]
        avg_l = res[strat]["avg_length"]
        print(f"  {strat:15}: count={cnt:2d}, avg_len={avg_l:6.1f}")

print("\n=== 2. SIMILARITY PREDICTIONS ===")
pairs = [
    ("Khách hàng có thể yêu cầu trả hàng trong vòng 30 ngày.", "Thời hạn đổi trả sản phẩm cho người mua tối đa là một tháng.", "cao"),
    ("Tiki hỗ trợ hoàn tiền toàn bộ giá trị đơn hàng cho khách hàng.", "Tiki từ chối hoàn tiền và hủy bỏ yêu cầu khiếu nại của khách hàng.", "trung bình"),
    ("Phương thức hoàn tiền qua thẻ tín dụng mất từ bảy đến mười lăm ngày.", "Mô hình ngôn ngữ lớn huấn luyện dựa trên kiến trúc Transformer.", "thấp"),
    ("Thời gian xử lý khiếu nại đổi trả cho khách hàng mua sắm là ba mươi ngày.", "Thời hạn tối đa để nhà bán phản hồi yêu cầu trả hàng là bốn mươi tám giờ.", "thấp"),
    ("Sản phẩm đổi trả phải còn nguyên tem niêm phong và chưa qua sử dụng.", "Hàng gửi hoàn bắt buộc giữ nguyên vẹn bao bì đóng gói và nhãn mác ban đầu.", "cao")
]

for idx, (a, b, pred) in enumerate(pairs, 1):
    va = _mock_embed(a)
    vb = _mock_embed(b)
    sim = compute_similarity(va, vb)
    print(f"Cặp {idx}: pred={pred:10s} | sim={sim:+.4f} | A: {a[:35]}... | B: {b[:35]}...")
