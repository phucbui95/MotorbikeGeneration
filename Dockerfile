FROM python:3.7-slim-buster

# Install gcc
RUN apt-get update
RUN apt-get install -y libpq-dev libc-dev gcc

# Install python libraries
COPY requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

RUN apt-get install -y postgresql-client

# Copy source code
COPY . /app
WORKDIR /app

ENTRYPOINT ["./run_all.sh"]