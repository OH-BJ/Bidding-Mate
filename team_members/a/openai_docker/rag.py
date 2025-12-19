from loader import load_documents
from chunker import chunk_text
from embedder import get_embeddings
from vectordb import VectorDB
from openai import OpenAI

# client = OpenAI()

class RAGPipeline:
    def __init__(self, db_path="vectordb.pkl"):
        self.db = VectorDB(db_path)
        self.client = OpenAI() # 여기서 생성하는 게 안전함

    def build(self, raw_dir="../../../data/raw"):
        print("📄 pdf 문서 로딩...")
        docs = load_documents(raw_dir)
        print("✂️ Chunking...")
        chunks = chunk_text(docs)
        print("🧠 Embedding...")
        print("🔧 임베딩 생성 시작")
        vectors = get_embeddings(chunks)
        print("🔧 임베딩 생성 완료")
        print("💾 Saving to VectorDB...")
        self.db.save(chunks, vectors)
        print("✅ RAG build complete")

    def query(self, question):
        print("🔍 Searching similar chunks...")
        # 1) 질문을 임베딩으로 변환 
        q_vec = get_embeddings([question])[0] 
        # 2) 벡터로 검색 
        top_chunks = self.db.search(q_vec, top_k=3)
        # 3) 프롬프트 구성 
        context = "\n\n".join(top_chunks)
        prompt = f"다음 내용을 참고해서 질문에 답변해줘:\n\n{context}\n\n질문: {question}"
        print("🤖 Generating answer...")
        # 4) LLM 호출
        completion = self.client.chat.completions.create(
            model="gpt-5-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        return completion.choices[0].message.content

'''
python3 - << 'EOF'
from rag import RAGPipeline

rag = RAGPipeline("testdb.pkl")
rag.build("../../../data/raw")

answer = rag.query("입찰 공고 조건 요약해줘")
print("답변:", answer)
EOF

'''
