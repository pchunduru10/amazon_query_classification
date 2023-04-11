# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.10-slim

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Install pip requirements
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

WORKDIR /xoi

COPY config.json .
COPY ./app ./app
COPY ./model ./model
COPY run.sh .

# CMD ["python", "./model/train_model.py"]
CMD ["sh", "./run.sh"]
