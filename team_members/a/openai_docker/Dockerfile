# FROM python:3.12-slim
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /work

# 1) requirements만 먼저 복사 
COPY requirements.txt . 
# 2) pip install 
RUN pip install --no-cache-dir -r requirements.txt 
# 3) 나머지 코드 전체 복사 
COPY . .

CMD ["python", "inference.py"]
