def chunk_text(docs, chunk_size=300):
    chunks = []
    for doc in docs:
        doc = doc.strip()
        if not doc:
            continue
        for i in range(0, len(doc), chunk_size):
            # chunks.append(doc[i:i+chunk_size])
            chunk = doc[i:i+chunk_size].strip()
            if chunk:
                chunks.append(chunk)

    return chunks

'''
python3 - << 'EOF'
from chunker import chunk_text
docs = ["이것은 테스트 문서입니다. " * 20]
chunks = chunk_text(docs, chunk_size=50)
print("청크 개수:", len(chunks))
print("첫 청크:", chunks[0])
EOF


python3 - << 'EOF'
from loader import load_documents
docs = load_documents("../../../data/raw")
print("문서 개수:", len(docs))
print("첫 문서 내용 일부:", docs[0][:200] if docs else "문서 없음")

from chunker import chunk_text
chunks = chunk_text(docs, chunk_size=50)
print("청크 개수:", len(chunks))
print("첫 청크:", chunks[0])
EOF

'''
