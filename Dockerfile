FROM vastai/pytorch

RUN apt-get update

RUN apt-get install -y libgtk2.0-dev
RUN apt-get install -y libsm6 libxext6 libxrender-dev

# Install python libraries
COPY requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

RUN apt-get install -y zip wget
RUN mkdir -p /root/.kaggle/
ADD kaggle.json /root/.kaggle/

# Copy source code
COPY . /app/
WORKDIR /app

#ENTRYPOINT ["/bin/bash"]

EXPOSE 22 8080