FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir --default-timeout=100 --retries=10 -r requirements.txt

COPY predict.py .

EXPOSE 8010

CMD ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "8020"]

