FROM python:3.11-slim

WORKDIR /app

COPY . /app/crisisops_env

RUN pip install --no-cache-dir -e /app/crisisops_env

EXPOSE 8000

CMD ["uvicorn", "crisisops_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
