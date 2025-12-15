FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 실행 명령어
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]