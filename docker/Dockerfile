FROM vastai/pytorch

RUN apt-get update

#RUN apt-get install -y libgtk2.0-dev
#RUN apt-get install -y libsm6 libxext6 libxrender-dev

# Install python libraries
COPY requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

# Copy model
COPY ./final_model /app/checkpoint
COPY ./scripts/predict.sh /model/predict.sh

RUN rm -rf keys
RUN rm -rf kaggle.json
RUN mkdir -p /app/outputs
RUN ln -fs /app/outputs /result
RUN ln -fs /data /app/data

# Copy source code
COPY . /app/
WORKDIR /app
