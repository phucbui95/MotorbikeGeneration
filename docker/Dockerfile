FROM vastai/pytorch

RUN apt-get update

RUN apt-get install -y libgtk2.0-dev
RUN apt-get install -y libsm6 libxext6 libxrender-dev

# Install python libraries
COPY requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

# Copy model
COPY ./data/models/exp2 /app/checkpoint
COPY ./data/resized128_image/resized128_image_fixed/label.csv /app/data/resized128_image/resized128_image_fixed/label.csv

# Copy source code
COPY . /app/
WORKDIR /app


EXPOSE 22 8080