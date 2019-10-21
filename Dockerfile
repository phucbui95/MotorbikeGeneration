FROM vastai/pytorch

RUN apt-get update
RUN apt-get install -y libsm6 libxext6 libxrender-dev

# Install python libraries
COPY requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

RUN apt-get install -y zip
RUN mkdir -p /root/.kaggle/
ADD kaggle.json /root/.kaggle/

# Copy source code
COPY . /app/MotorbikeGeneration
WORKDIR /app

ENTRYPOINT ["/bin/bash"]