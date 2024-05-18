FROM python:3.9-slim

RUN pip install -U pip

COPY requirements.txt ./tmp/requirements.txt

RUN pip install --upgrade -r ./tmp/requirements.txt && python -m spacy download en
