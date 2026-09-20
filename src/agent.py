from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        results = self.store.search(question, top_k=top_k)
        if not results:
            return "Không tìm thấy thông tin phù hợp trong cơ sở dữ liệu để trả lời câu hỏi."

        context_blocks = []
        for idx, res in enumerate(results, start=1):
            source = res.get("metadata", {}).get("source", res.get("id", f"doc_{idx}"))
            context_blocks.append(f"[{idx}] (Nguồn: {source})\n{res['content']}")

        context_text = "\n\n".join(context_blocks)
        prompt = (
            f"Bạn là trợ lý AI trả lời câu hỏi dựa trên cơ sở tri thức.\n"
            f"Hãy trả lời câu hỏi dưới đây một cách chính xác dựa trên ngữ cảnh được cung cấp. "
            f"Trích dẫn số thứ tự nguồn [1], [2],... khi đưa ra thông tin. "
            f"Nếu ngữ cảnh không chứa thông tin để trả lời, hãy nói rõ là không tìm thấy.\n\n"
            f"Ngữ cảnh:\n{context_text}\n\n"
            f"Câu hỏi: {question}\n\n"
            f"Câu trả lời:"
        )
        return self.llm_fn(prompt)
