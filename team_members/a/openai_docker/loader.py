import os
import sys
# import pdfplumber
import fitz # PyMuPDF

# PyMuPDF 경고 메시지 숨기기 
devnull = open(os.devnull, 'w') 
sys.stderr = devnull

def load_documents(raw_dir):
    docs = []
    for filename in os.listdir(raw_dir):
        if not filename.lower().endswith(".pdf"):
            continue

        path = os.path.join(raw_dir, filename)
        print(f"📄 Loading PDF: {filename}") 
        
        text = "" 
        # with pdfplumber.open(path) as pdf: 
        #     for page in pdf.pages: 
        #         extracted = page.extract_text() 
        #         if extracted: 
        #             text += extracted + "\n" 
        # if not text.strip(): 
        #     print(f"⚠️ 텍스트 없음: {filename}") 
        #     continue

        # docs.append(text)
        try:
            doc = fitz.open(path)
        except Exception as e:
            print(f"❌ PDF 열기 실패: {filename} ({e})") 
            continue

        # 2) 페이지 읽기 예외 처리 
        for page_number, page in enumerate(doc):
        # for page in doc:
            try:
                text += page.get_text() + "\n"
            except Exception as e: 
                print(f"⚠️ 페이지 읽기 실패: {filename} ({e})") 
                continue

        doc.close()

        if text.strip():
            docs.append(text)
        else:
            print(f"⚠️ 텍스트 없음: {filename}")
            
    return docs

'''
python3 - << 'EOF'
from loader import load_documents
docs = load_documents("../../../data/raw")
print("문서 개수:", len(docs))
print("첫 문서 내용 일부:", docs[0][:200] if docs else "문서 없음")
EOF
'''
