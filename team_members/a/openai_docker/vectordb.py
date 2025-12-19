import pickle
import numpy as np

class VectorDB:
    def __init__(self, path="vectordb.pkl"):
        self.path = path
        self.chunks = []
        self.vectors = []

        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
                self.chunks = data["chunks"]
                self.vectors = data["vectors"]
        except:
            pass

    def save(self, chunks, vectors):
        self.chunks = chunks
        self.vectors = vectors
        with open(self.path, "wb") as f:
            pickle.dump({"chunks": chunks, "vectors": vectors}, f)

    def search(self, query_vector, top_k=3): 
        q = np.array(query_vector) 
        sims = [] 
        for i, v in enumerate(self.vectors): 
            v = np.array(v) 
            sim = np.dot(q, v) / (np.linalg.norm(q) * np.linalg.norm(v)) 
            sims.append((sim, self.chunks[i])) 
            
        sims.sort(reverse=True, key=lambda x: x[0]) 
        return [c for _, c in sims[:top_k]]

'''
python3 - << 'EOF'
from vectordb import VectorDB
from embedder import get_embeddings

chunks = ["테스트 문장입니다.", "두 번째 문장입니다."]
vectors = get_embeddings(chunks)

db = VectorDB("testdb.pkl")
db.save(chunks, vectors)

query = "테스트"
query_vec = get_embeddings([query])[0]

result = db.search(query_vec, top_k=1)
print(result)
EOF

'''

