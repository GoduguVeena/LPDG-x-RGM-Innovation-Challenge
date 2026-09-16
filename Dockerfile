FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY baseline_3sigma.py .
COPY validate_submission.py .
COPY src ./src
COPY models ./models

CMD ["python", "src/final_predict.py"]