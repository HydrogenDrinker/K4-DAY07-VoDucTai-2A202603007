"""
Benchmark Script for Lab 07 — Data Foundations & Vector Retrieval
Strategy: SentenceChunker (max_sentences_per_chunk=3)
Student: Vo Duc Tai - MSSV: 2A202603007
Dataset: Tiki E-commerce Return Policy (data/ecommerce/)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Fix stdout encoding for Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.agent import KnowledgeBaseAgent
from src.chunking import SentenceChunker
from src.embeddings import _mock_embed
from src.models import Document
from src.store import EmbeddingStore


DATA_DIR = Path("data/ecommerce")

BENCHMARK_QUERIES = [
    {
        "id": 1,
        "query": "Thời hạn tối đa để phản hồi và xử lý khiếu nại đổi trả là bao lâu?",
        "filter": {"audience": "buyer"},
        "gold": "Tối đa 30 ngày đối với Tiki Trading và 07 ngày đối với Nhà Bán Hàng đối tác (TikiNGON trong 02h-24h).",
        "gold_doc": "buyer-return-refund-period",
        "description": "Câu hỏi cần lọc audience=buyer để tránh lẫn thời hạn 48h của người bán",
    },
    {
        "id": 2,
        "query": "Khách hàng cần đáp ứng những điều kiện gì về tình trạng sản phẩm để được chấp nhận đổi trả tại Tiki?",
        "filter": None,
        "gold": "Sản phẩm còn nguyên tem bảo hành, nguyên mã vạch, chưa qua sử dụng, chưa kích hoạt bảo hành điện tử, đầy đủ hộp đựng và phụ kiện, kèm video mở hộp.",
        "gold_doc": "buyer-return-conditions",
        "description": "Điều kiện và lý do chấp nhận đổi trả cho khách hàng",
    },
    {
        "id": 3,
        "query": "Nếu thanh toán bằng thẻ tín dụng quốc tế Visa hoặc Mastercard thì thời gian hoàn tiền là bao nhiêu ngày làm việc?",
        "filter": None,
        "gold": "Từ 07 đến 15 ngày làm việc tùy theo quy trình đối soát và chu kỳ sao kê của ngân hàng phát hành thẻ.",
        "gold_doc": "buyer-refund-timeline-methods",
        "description": "Thời gian hoàn tiền cho thẻ tín dụng quốc tế",
    },
    {
        "id": 4,
        "query": "Những mặt hàng nào thuộc danh mục không áp dụng chính sách đổi trả vì lý do đổi ý tại Tiki?",
        "filter": None,
        "gold": "Thẻ cào/e-voucher kỹ thuật số, thực phẩm đã mở nắp/seal, đồ lót/đồ bơi, mỹ phẩm đã rách màng bọc nilon, hàng may đo in ấn cá nhân hóa, hàng thanh lý xả kho.",
        "gold_doc": "non-returnable-categories",
        "description": "Danh mục hàng hóa không áp dụng đổi trả",
    },
    {
        "id": 5,
        "query": "Nhà bán hàng cần gửi khiếu nại bồi thường trong vòng bao lâu nếu hàng trả về bị tráo đổi hoặc hư hỏng nặng?",
        "filter": None,
        "gold": "Trong vòng 03 ngày làm việc kể từ thời điểm nhận hàng hoàn, kèm video đóng gói xuất kho và video mở kiện hàng hoàn trả.",
        "gold_doc": "seller-return-dispute-compensation",
        "description": "Thời hạn và điều kiện khiếu nại bồi thường cho Nhà bán",
    },
]


def parse_markdown_file(path: Path) -> tuple[dict, str]:
    raw = path.read_text(encoding="utf-8")
    if raw.startswith("---"):
        parts = raw.split("---", 2)
        if len(parts) >= 3:
            fm_text = parts[1]
            body = parts[2].strip()
            metadata = dict(re.findall(r"^(\w+):\s*(.+)$", fm_text, re.M))
            return metadata, body
    return {}, raw.strip()


def mock_llm(prompt: str) -> str:
    """Simple concise extractive answer generator based on context."""
    # Find sentences in prompt that match key query keywords
    return (
        "Dựa trên tài liệu được cung cấp, câu trả lời được xác định từ các điều khoản quy định. "
        "Chi tiết cụ thể xem trong phần trích dẫn nguồn kèm theo."
    )


def run_benchmark():
    print("=" * 70)
    print("LAB 07 BENCHMARK — STRATEGY: SentenceChunker(max_sentences_per_chunk=3)")
    print("Student: Vo Duc Tai — MSSV: 2A202603007")
    print(f"Dataset Directory: {DATA_DIR.resolve()}")
    print("=" * 70)

    chunker = SentenceChunker(max_sentences_per_chunk=3)
    store = EmbeddingStore(collection_name="tiki_policy_benchmark", embedding_fn=_mock_embed)
    agent = KnowledgeBaseAgent(store=store, llm_fn=mock_llm)

    all_docs: list[Document] = []
    file_paths = sorted(DATA_DIR.glob("*.md"))

    total_characters = 0
    print("\n[1] Nạp và phân đoạn (Chunking) tài liệu:")
    for path in file_paths:
        metadata, body = parse_markdown_file(path)
        doc_id = metadata.get("doc_id", path.stem)
        total_characters += len(body)
        chunks = chunker.chunk(body)

        print(f"  - File: {path.name:38} | Chunks: {len(chunks):2d} | Audience: {metadata.get('audience', 'N/A')}")
        for i, chunk_text in enumerate(chunks):
            chunk_doc = Document(
                id=f"{doc_id}#{i}",
                content=chunk_text,
                metadata={
                    **metadata,
                    "doc_id": doc_id,
                    "chunk_index": i,
                    "source": str(path),
                },
            )
            all_docs.append(chunk_doc)

    store.add_documents(all_docs)
    print(f"\n=> Đã nạp thành công {len(all_docs)} chunks từ {len(file_paths)} tài liệu ({total_characters} ký tự) vào store.\n")

    print("=" * 70)
    print("[2] ĐÁNH GIÁ 5 BENCHMARK QUERIES:")
    print("=" * 70)

    for q in BENCHMARK_QUERIES:
        qid = q["id"]
        query = q["query"]
        meta_filter = q["filter"]
        gold = q["gold"]
        gold_doc = q["gold_doc"]

        print(f"\n--- Câu hỏi #{qid}: {query}")
        print(f"    Bộ lọc metadata: {meta_filter}")
        print(f"    Tài liệu chứa đáp án chuẩn (Gold Doc): {gold_doc}")
        print(f"    Câu trả lời chuẩn (Gold Answer): {gold}")

        results = store.search_with_filter(query, top_k=3, metadata_filter=meta_filter)
        print(f"    Top-3 chunks truy xuất được:")
        for rank, res in enumerate(results, start=1):
            source_id = res.get("metadata", {}).get("doc_id")
            chunk_id = res.get("id")
            score = res.get("score", 0.0)
            is_gold = " [GOLD]" if source_id == gold_doc else ""
            preview = res.get("content", "").replace("\n", " ")[:120]
            print(f"      {rank}. [Score: {score:.4f}] Doc: {chunk_id}{is_gold}")
            print(f"         Nội dung: {preview}...")

    # A/B Test for Query 1
    print("\n" + "=" * 70)
    print("[3] THỬ NGHIỆM A/B TRÊN CÂU HỎI #1: CÓ vs KHÔNG CÓ METADATA FILTER")
    print("=" * 70)
    q1 = BENCHMARK_QUERIES[0]["query"]
    print(f"Query: \"{q1}\"\n")

    res_with_filter = store.search_with_filter(q1, top_k=3, metadata_filter={"audience": "buyer"})
    print("A. CÓ METADATA FILTER (audience=buyer):")
    for rank, r in enumerate(res_with_filter, start=1):
        print(f"   {rank}. {r['id']} (audience: {r['metadata'].get('audience')}) score={r['score']:.4f}")
        print(f"      {r['content'][:90].replace(chr(10), ' ')}...")

    res_no_filter = store.search_with_filter(q1, top_k=3, metadata_filter=None)
    print("\nB. KHÔNG CÓ METADATA FILTER (Không lọc đối tượng):")
    for rank, r in enumerate(res_no_filter, start=1):
        print(f"   {rank}. {r['id']} (audience: {r['metadata'].get('audience')}) score={r['score']:.4f}")
        print(f"      {r['content'][:90].replace(chr(10), ' ')}...")


if __name__ == "__main__":
    import io
    old_stdout = sys.stdout
    buffer = io.StringIO()
    sys.stdout = buffer
    run_benchmark()
    output_text = buffer.getvalue()
    sys.stdout = old_stdout
    print(output_text)
    Path("ket_qua_benchmark.txt").write_text(output_text, encoding="utf-8")

