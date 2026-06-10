FROM python:3.11-slim

WORKDIR /app

COPY gapsight-backend/requirements.txt ./backend_reqs.txt
COPY gapsight-frontend/requirements.txt ./frontend_reqs.txt
RUN pip install --no-cache-dir -r backend_reqs.txt -r frontend_reqs.txt

COPY gapsight-backend ./gapsight-backend
COPY gapsight-frontend ./gapsight-frontend

EXPOSE 7860 8000

WORKDIR /app/gapsight-backend
CMD sh -c "uvicorn app.main:app --host 0.0.0.0 --port 8000 & cd /app/gapsight-frontend && streamlit run app.py --server.port ${PORT:-7860} --server.address 0.0.0.0"
