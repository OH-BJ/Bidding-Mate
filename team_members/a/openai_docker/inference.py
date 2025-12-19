from rag import RAGPipeline

def main():
    print("inference start")

    rag = RAGPipeline()

    # 최초 1회만 실행 (벡터DB 생성)
    # rag.build()
    rag.build("/work/data/raw")

    question = "입찰 공고 조건 요약해줘"
    answer = rag.query(question)

    print("답변:", answer)

if __name__ == "__main__":
    main()
