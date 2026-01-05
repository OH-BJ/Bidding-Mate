# Bidding-Mate: 입찰 공고 분석 AI 에이전트

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python"/>
  <img src="https://img.shields.io/badge/LangChain-v0.1-green?logo=langchain"/>
  <img src="https://img.shields.io/badge/LangGraph-Agent-orange"/>
  <img src="https://img.shields.io/badge/Streamlit-App-red?logo=streamlit"/>
  <img src="https://img.shields.io/badge/Docker-Container-2496ED?logo=docker&logoColor=white"/>
</div>

<br>

## 프로젝트 소개 (Overview)
**Bidding-Mate**는 평균 7만 자에 달하는 방대한 공공 입찰 공고문을 분석하여, 사용자가 원하는 핵심 정보(예산, 기간, 평가 기준 등)를 정확하게 추출해주는 **RAG(검색 증강 생성) 기반 AI 에이전트**입니다.

단순한 키워드 검색을 넘어, **LangGraph**를 도입하여 사용자의 의도를 파악하고 답변의 품질을 스스로 검증하는 **순환형(Cyclic) 아키텍처**를 구현했습니다.

### 핵심 문제 정의 (Problem Statement)
* **정보 과부하:** 공고문 하나당 평균 7.2만 자, 최대 23만 자의 방대한 텍스트.
* **비효율성:** 입찰 담당자가 핵심 정보(배점, 자격 요건)를 찾는 데 과도한 시간 소요.
* **LLM의 한계:** 범용 모델(GPT-4 등) 사용 시, 구체적인 팩트(예산, 날짜)에 대한 **할루시네이션(거짓 답변)** 발생.

---

## 시스템 아키텍처 (Architecture)

본 프로젝트는 선형적인 RAG 구조를 탈피하고, **LangGraph**를 활용한 제어 가능한(Controllable) 에이전트 파이프라인을 구축했습니다.

### Workflow Steps
1.  **Router (의도 파악):** 사용자의 질문이 '입찰 분석'인지 '단순 인사'인지 분류하여 불필요한 검색 비용 절감.
2.  **Retriever (문서 검색):** **MMR(Maximal Marginal Relevance)** 알고리즘을 통해 질문과 관련되면서도 다양한 정보를 탐색.
3.  **Grader (품질 평가):** 검색된 문서가 질문에 답변하기 적절한지 LLM이 스스로 채점(Grading).
    * 적합(Yes) → 답변 생성 단계로 이동
    * 부적합(No) → **Fallback** (답변 불가 선언 또는 재검색)
4.  **Generator (답변 생성):** 검증된 문서만을 바탕으로 최종 답변 생성 (Hallucination 방지 프롬프트 적용).

---

## 코드 구조 (Code Structure: OOP & Modularity)
본 프로젝트는 유지보수성과 확장성을 고려하여 **객체 지향 프로그래밍(OOP)** 원칙에 따라 설계되었습니다.

* **Modularity:** `Router`, `Retriever`, `Grader`, `Generator` 등 각 에이전트 기능을 독립적인 메서드와 클래스로 분리하여 관리.
* **Scalability:** LangGraph의 `StateGraph` 클래스를 상속받아 워크플로우를 객체로 정의함으로써, 추후 새로운 노드(기능) 추가가 용이함.
* **Abstraction:** 복잡한 RAG 로직을 `BiddingAgent` 클래스 내부로 캡슐화하여, 메인 실행 파일(`app.py`)의 코드를 간결하게 유지.

---

## 기술 스택 (Tech Stack)

* **LLM:** OpenAI GPT-5 (Generator), GPT-5-mini (Router/Grader)
* **Embedding:** OpenAI `text-embedding-3-small`
* **Vector DB:** ChromaDB
* **Framework:** LangChain, LangGraph
* **DevOps:** Docker, Docker Compose
* **Evaluation:** Ragas
* **UI/UX:** Streamlit

---

## 설치 및 실행

### Docker로 실행하기
복잡한 환경 설정 없이 도커만 있으면 즉시 실행 가능합니다.

#### 1. 리포지토리 클론
```bash
git clone https://github.com/OH-BJ/Bidding-Mate.git
cd Bidding-Mate
```

#### 2. 데이터 폴더 배치
프로젝트 루트 경로에 `data` 폴더가 위치해야 합니다.
제공된 PDF 파일들이 아래 경로 구조와 정확히 일치하는지 확인해 주세요.

```text
Bidding-Mate/
└── data/
    └── raw/
        └── 100_PDF/
            ├── 공고문1.pdf
            ├── 공고문2.pdf
            └── ... (나머지 PDF 파일들)
```

#### 3. 환경 변수 설정
프로젝트 루트 경로에 `.env` 파일을 생성하고, 발급받은 OpenAI API 키를 입력하세요.

```env
OPENAI_API_KEY=sk-proj-xxxxxxxx...
```

#### 4. 컨테이너 빌드 및 실행
터미널에서 아래 명령어를 입력하여 도커 컨테이너를 실행합니다.

```bash
docker-compose up --build
```
참고: 처음 실행 시에는 Vector DB가 생성되지 않아 웹 페이지에 에러가 표시될 수 있습니다. 이는 정상적인 현상이므로 당황하지 말고 **다음 단계(DB 생성)**를 진행해 주세요.

#### 5. Vector DB 생성
**새로운 터미널 창**을 열고, 실행 중인 도커 컨테이너 내부에서 DB 생성 스크립트를 작동시킵니다.

```bash
# 실행 중인 'app' 컨테이너에 접속하여 DB 생성 스크립트 실행
docker-compose exec app python db_maker.py
```
주의: PDF 데이터 양에 따라 생성 시간이 소요될 수 있습니다. 터미널에 완료 메시지가 뜰 때까지 기다려주세요.

#### 6. 웹 접속 및 확인
브라우저 주소창에 아래 URL을 입력하여 접속합니다.

**http://localhost:8501**

* 좌측 사이드바에 **"System Status: Online"** 초록색 배지가 표시되면 설치가 성공적으로 완료된 것입니다.

## 프로젝트 산출물 (Project Deliverables)

### 1. 최종 보고서 (Project Report)
프로젝트의 진행 과정, 상세 아키텍처, 성능 실험 결과 및 개선 사항을 정리한 최종 보고서입니다.
> **[Bidding-Mate_최종보고서.pdf 다운로드](./assets/Bidding-Mate_Final_Report.pdf)**

### 2. 협업 일지 (Collaboration Logs)
팀원별 개발 과정과 회고가 담긴 협업 일지입니다.

| 이름 | 협업 일지 링크 |
| :--- | :--- | :--- |
| [오병주] | [블로그/노션 링크](https://www.notion.so/1-15-2df657925cde80a6b81fd8ab6c1fcc47?source=copy_link) |
| [김소희] | [블로그/노션 링크](https://team1-blog-url.com) |
| [손성경] | [블로그/노션 링크](https://team1-blog-url.com) |
| [신아름] | [블로그/노션 링크](https://team3-blog-url.com) |
| [최지혁] | [블로그/노션 링크](https://team3-blog-url.com) |