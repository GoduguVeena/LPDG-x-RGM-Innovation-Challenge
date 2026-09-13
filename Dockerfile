FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY baseline_3sigma.py .
COPY validate_submission.py .

CMD ["sh", "-c", "python baseline_3sigma.py --data /app/data --out /app/outputs/predictions.csv && python validate_submission.py /app/outputs/predictions.csv"]