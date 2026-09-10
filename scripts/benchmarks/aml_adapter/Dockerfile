FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app/scripts/benchmarks/aml_adapter

ENV PYTHONPATH=/app
EXPOSE 8090

CMD ["sh", "-c", "exec gunicorn --bind 0.0.0.0:${PORT:-8090} --workers ${AML_ADAPTER_WORKERS:-4} --threads ${AML_ADAPTER_THREADS:-8} --timeout ${AML_ADAPTER_GUNICORN_TIMEOUT_SECONDS:-45} scripts.benchmarks.aml_adapter.app:app"]
