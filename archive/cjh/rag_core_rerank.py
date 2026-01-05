import os, re
from typing import TypedDict, List, Dict, Any
from dotenv import load_dotenv

# LangChain 관련 임포트
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END

# 분리한 prompt.py에서 프롬프트 객체들 임포트
from archive.cjh.prompt_rerank import ROUTER_PROMPT, GRADER_PROMPT, GENERATOR_PROMPT, RERANK_PROMPT

# 환경변수 로드
load_dotenv()

class BiddingAgent:
    def __init__(self, db_path="./chroma_db_chunk500", model_heavy="gpt-5", model_light="gpt-5-mini"):
        """
        초기화: DB 로드, LLM 설정(Heavy & Light), 그래프(Workflow) 빌드
        """
        # 모델 이원화
        self.llm_heavy = ChatOpenAI(model=model_heavy, temperature=0)
        self.llm_light = ChatOpenAI(model=model_light, temperature=0)
        
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        # DB 연결
        if not os.path.exists(db_path):
            print(f"경고: {db_path}를 찾을 수 없습니다. 현재 위치: {os.getcwd()}")
            
        self.vectorstore = Chroma(persist_directory=db_path, embedding_function=self.embeddings)
        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 20,
                "fetch_k" : 50,
                "lambda_mult": 0.85 
            } 
        )
        
        self.app_workflow = self._build_graph()

    class GraphState(TypedDict):
        question: str
        context: List[Dict[str, Any]]
        answer: str
        router_ok: bool
        doc_ok: bool
        rerank_ok: bool

    # 문서를 보기 좋게 꾸미는 함수 (메타데이터 활용)
    def _format_docs(self, docs: List[Dict[str, Any]]) -> str:
        formatted_docs = []
        for i, doc in enumerate(docs):
            # 딕셔너리에서 메타데이터 안전하게 꺼내기
            project_name = doc.get("project_name", "정보없음")
            budget = doc.get("budget", "정보없음")
            notice_no = doc.get("notice_no", "정보없음")
            agency = doc.get("agency", "정보없음")
            content = doc.get("content", "")
            
            # AI에게 보여줄 포맷 구성
            enriched_content = (
                f"[참고문서 {i+1}]\n"
                f"- 공고번호: {notice_no}\n"
                f"- 사업명: {project_name}\n"
                f"- 발주기관: {agency}\n"
                f"- 확정예산(CSV): {budget}\n" # CSV 정답을 직접 노출
                f"내용:\n{content}"
            )
            formatted_docs.append(enriched_content)
            
        return "\n\n".join(formatted_docs)

    def _route_question(self, state):
        print(f"---[1] 의도 파악 중 (Light Model): {state['question']}---")
        
        chain = ROUTER_PROMPT | self.llm_light | StrOutputParser()
        category = chain.invoke({"question": state['question']}).lower()
        
        return {"router_ok": category.strip() == "bid"}

    def _retrieve(self, state):
        print(f"---[2] 문서 검색 중: {state['question']}---")
        docs = self.retriever.invoke(state['question'])
        
        context = []
        for doc in docs:
            # DB에서 꺼낼 때 메타데이터도 함께 딕셔너리에 담기
            context.append({
                "content": doc.page_content,
                "source": doc.metadata.get("source", "출처 미상"),
                "project_name": doc.metadata.get("project_name", "정보없음"),
                "budget": doc.metadata.get("budget", "정보없음"),
                "notice_no": doc.metadata.get("notice_no", "정보없음"),
                "agency": doc.metadata.get("agency", "정보없음")
            })
            
        return {"context": context}

    def _grade_documents(self, state):
        print(f"---[3] 문서 품질 채점 중 (Light Model)---")
        
        question = state['question']
        docs = state['context']
        
        if not docs:
            return {"doc_ok": False, "rerank_ok": False}
        
        # 단순 텍스트 결합 대신 _format_docs 사용하여 메타데이터 포함
        # 상위 10개만 검사
        doc_sample = self._format_docs(docs[:10])
        
        chain = GRADER_PROMPT | self.llm_light | StrOutputParser()
        score = chain.invoke({"question": question, "context": doc_sample}).lower()
        
        is_relevant = "yes" in score
        
        if is_relevant:
            print(" -> 관련성 있음 (통과)")
        else:
            print(" -> 관련성 없음 (탈락)")
            
        return {"doc_ok": is_relevant, "rerank_ok": False}
    
    def _rerank_documents(self, state):
        print(f"---[3.5] 문서 rerank 중 (Light Model)---")

        question = state["question"]
        docs = state["context"]
        if not docs:
            return {"rerank_ok": False, "context": []}

        chain = RERANK_PROMPT | self.llm_light | StrOutputParser()
        scored = []

        # 비용 때문에 상위 일부만 rerank 권장
        for d in docs[:10]:
            doc_text = self._format_docs([d])
            raw = chain.invoke({"question": question, "context": doc_text}).strip()
            
            m = re.search(r"[-+]?\d*\.?\d+", raw)
            score = float(m.group()) if m else 0.0
            
            scored.append((score, d))
            
        if not scored:
            return {"rerank_ok": False, "context": []}

        scored.sort(key=lambda x: x[0], reverse=True)

        TOP_N = 5
        top_docs = [d for s, d in scored[:TOP_N]]

        rerank_ok = len(top_docs) > 0 and scored[0][0] >= 1.0  # 임계값은 조절
        return {"rerank_ok": rerank_ok, "context": top_docs}

    def _generate(self, state):
        print(f"---[4] 최종 답변 생성 중 (Heavy Model)---")
        question = state['question']
        
        # _format_docs 사용하여 예산/사업명 정보가 포함된 텍스트 전달
        context_text = self._format_docs(state['context'])
        
        chain = GENERATOR_PROMPT | self.llm_heavy | StrOutputParser()
        response = chain.invoke({"context": context_text, "question": question})
        
        return {"answer": response}

    def _rewrite_query(self, state):
        return {"answer": "죄송합니다. 저는 공고문 분석 전문가로서 사업 및 입찰과 관련된 질문에만 답변을 드릴 수 있습니다.\n(또는 관련 문서를 찾지 못했습니다.)"}

    def _build_graph(self):
        workflow = StateGraph(self.GraphState)
        
        workflow.add_node("router", self._route_question) 
        workflow.add_node("retrieve", self._retrieve)
        workflow.add_node("grade", self._grade_documents)
        workflow.add_node("generate", self._generate)
        workflow.add_node("fallback", self._rewrite_query)
        workflow.add_node("rerank", self._rerank_documents)
        
        workflow.set_entry_point("router")
        
        workflow.add_conditional_edges(
            "router",
            lambda x: "retrieve" if x["router_ok"] else "fallback",
            {"retrieve": "retrieve", "fallback": "fallback"}
        )
        
        workflow.add_edge("retrieve", "grade")
        
        workflow.add_conditional_edges(
            "grade",
            lambda x: "rerank" if x["doc_ok"] else "fallback",
            {"rerank": "rerank", "fallback": "fallback"}
        )

        workflow.add_conditional_edges(
            "rerank",
            lambda x: "generate" if x.get("rerank_ok", False) else "fallback",
            {"generate": "generate", "fallback": "fallback"}
        )
        
        workflow.add_edge("generate", END)
        workflow.add_edge("fallback", END)
        
        return workflow.compile()

    def get_answer(self, question: str):
        inputs = {"question": question}
        result = self.app_workflow.invoke(inputs)
        
        answer = result.get('answer', '')
        
        if (not result.get("router_ok", True)) or \
           (not result.get("doc_ok", True)) or \
           (not result.get("rerank_ok", True)) or \
           "죄송합니다" in answer:
            return answer, [] 
            
        return answer, result.get('context', [])
    
    def ask_with_context(self, question):
        answer, contexts = self.get_answer(question)
        # API 반환용으로는 content만 간략히 리스트로 줌
        context_texts = [doc['content'] for doc in contexts] if contexts else []
        return {
            "question": question,
            "answer": answer,
            "contexts": context_texts 
        }