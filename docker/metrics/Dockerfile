FROM python:3-slim-buster

WORKDIR /data
COPY provisioning/requirements.txt .
RUN pip3 install -r /data/requirements.txt

WORKDIR /app

CMD export PYTHONPATH=/app; python3 main.py
