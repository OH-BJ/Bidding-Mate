from openai import OpenAI
from typing_extensions import override
from dotenv import load_dotenv

load_dotenv() # .env 파일 로드


client = OpenAI()

def get_embeddings(chunks):
    vectors = []
    for chunk in chunks:
        emb = client.embeddings.create(
            model="text-embedding-3-small",
            input=chunk
        ).data[0].embedding
        vectors.append(emb)
    return vectors
# from openai import OpenAI
# from dotenv import load_dotenv
# import os

# load_dotenv()  # .env 파일 로드

# client = OpenAI()

# def get_embeddings(texts, model="text-embedding-3-small"):
#     response = client.embeddings.create(
#         model=model,
#         input=texts
#     )
#     return [item.embedding for item in response.data]

'''
python3 - << 'EOF'
from embedder import get_embeddings
chunks = ["테스트 문장입니다."]
vectors = get_embeddings(chunks)
print("임베딩 길이:", len(vectors[0]))
EOF

'''
