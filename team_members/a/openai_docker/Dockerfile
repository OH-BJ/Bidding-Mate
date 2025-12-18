# FROM python:3.12-slim
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /work

# 1) requirements만 먼저 복사 
# COPY requirements.txt . 
COPY team_members/a/openai_docker/requirements.txt .
# 2) pip install 
RUN pip install --no-cache-dir -r requirements.txt 

# httpx 버전 충돌 해결
RUN pip install --upgrade httpx
RUN pip install --upgrade openai

# 3) 나머지 코드 전체 복사 
# COPY . .
# openai_docker 폴더 전체를 복사
COPY team_members/a/openai_docker /work
COPY .env /work/.env
COPY data /work/data

CMD ["python", "inference.py"]
