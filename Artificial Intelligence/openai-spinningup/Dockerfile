FROM ubuntu
COPY . /openai-spinningup

RUN apt update
RUN apt install python3 python3-pip python-opengl libopenmpi-dev xvfb -y

RUN pip3 install -r requirements.txt
