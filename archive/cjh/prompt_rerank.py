# prompt.py
from langchain_core.prompts import ChatPromptTemplate

# 1. 라우터 (Router) 프롬프트
# 질문이 사업 관련인지 잡담인지 판단
router_template_str = """
당신은 사용자의 질문을 분석하여 '입찰 공고 분석(bid)'과 관련이 있는지 판단하는 분류기입니다.

[분류 기준: 'bid'로 분류해야 하는 경우]
1. 입찰 공고의 예산, 기간, 자격 요건, 평가 기준 등 세부 정보를 묻는 경우
2. 특정 사업명, 공고 번호, 발주 기관(예: 고려대, 서울시 등)에 대한 검색을 요청하는 경우
3. 공고 내용의 요약, 정리, 브리핑을 요청하는 경우
4. 입찰 서류 작성에 필요한 정보를 묻는 경우

[분류 기준: 'not_relevant'로 분류해야 하는 경우]
1. 일상적인 인사(안녕, 반가워)
2. 입찰과 전혀 관련 없는 일반 상식 질문 (오늘 날씨 어때? 등)
3. 코딩 질문이나 시스템 자체에 대한 질문

질문이 '입찰 공고 분석'과 조금이라도 관련이 있다면 주저하지 말고 'bid'를 출력하세요.
오직 'bid' 또는 'not_relevant' 중 하나만 단어로 출력하세요.

질문: {question}
분류:
"""

ROUTER_PROMPT = ChatPromptTemplate.from_template(router_template_str)

# 2. 문서 품질 평가 (Grader) 프롬프트
# 검색된 문서가 질문에 적합한지 OX 퀴즈
grader_template_str = """
당신은 검색된 문서(Context)가 사용자의 질문(Question)에 답변하는 데 적합한지 평가하는 채점관입니다.
문서 내용이 질문과 조금이라도 관련이 있거나, 키워드가 포함되어 있다면 'yes'라고 답하세요.
엄격하게 평가하지 말고, 정보가 섞여 있어도 유용한 부분이 있다면 'yes'입니다.

질문: {question}
문서: {context}

이 문서가 답변에 도움이 됩니까? (yes 또는 no):
"""
GRADER_PROMPT = ChatPromptTemplate.from_template(grader_template_str)

# 3. 최종 답변 생성 (Generator) 프롬프트
# 문서를 보고 친절하게 답변 생성
generator_template_str = """
당신은 공고문 분석을 도와주는 친절한 AI 어시스턴트입니다.
아래 제공된 [참고 문서]를 바탕으로 사용자의 질문에 답변해 주세요.

[참고 문서]
{context}

[지시사항]
1. 반드시 제공된 [참고 문서]의 내용에 기반해서 답변하세요.
2. 문서에 없는 내용은 "문서에 해당 정보가 명시되어 있지 않습니다"라고 솔직하게 말하세요.
3. 예산이나 기간 같은 핵심 정보는 볼드체(**)로 강조해 주세요.
4. 답변은 간결하고 명확하게 작성해 주세요.
5. 만약 문서 헤더에 [확정예산(CSV)] 같은 정확한 정보가 있다면, 본문보다 우선해서 답변하세요.

질문: {question}
답변:
"""
GENERATOR_PROMPT = ChatPromptTemplate.from_template(generator_template_str)

# 4. 리랭크 프롬프트
rerank_template_str = """
너는 공고문 검색 결과를 재정렬하는 평가자다.

아래 [문서]가 사용자의 질문에 얼마나 관련 있는지
0점부터 10점 사이의 숫자로만 평가하라.
다른 텍스트 금지.

[평가 기준]
- 0점: 거의 무관함
- 5점: 부분적으로 관련 있음
- 10점: 질문에 직접적으로 답이 됨

[문서]
{context}

질문: {question}

출력 형식:
숫자 하나만 출력 (예: 7.5)
"""
RERANK_PROMPT = ChatPromptTemplate.from_template(rerank_template_str)