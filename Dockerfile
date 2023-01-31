FROM ubuntu:20.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install -y \
    python3 \
    python3-pip \
    libcairo2-dev \ 
    pkg-config \
    && apt-get update && apt-get install -y \
    vim \
    net-tools \
    openssh-server \ 
    libgirepository1.0-dev 

COPY ./app /app/
# EXPOSE 7860
RUN pip install -r /app/requirements.txt
CMD [ "python3", "/app/api_test.py" ]