FROM ubuntu
COPY . /openai-spinningup

RUN apt update
RUN apt install python3 python3-pip python-opengl xvfb -y

RUN pip3 install -r requirements.txt

CMD python3 /openai-spinningup/algs/run.py
