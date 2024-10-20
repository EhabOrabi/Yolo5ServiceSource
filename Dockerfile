FROM ultralytics/yolov5:latest-cpu
WORKDIR /usr/src/app
RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install -r requirements.txt

RUN apt-get update && \
    apt-get install -y procps && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


RUN curl -L https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5s.pt -o yolov5s.pt

COPY . .

ENV BUCKET_NAME="ehaborabi-bucket"
#ENV TELEGRAM_APP_URL="https://ehabo-polybot-k8s-v1.int-devops.click"
#ENV SQS_QUEUE_URL="https://sqs.eu-west-3.amazonaws.com/019273956931/ehabo-PolybotServiceQueue-k8s"
#ENV SQS_QUEUE_NAME="ehabo-PolybotServiceQueue-k8s"
ENV REGION_NAME="eu-west-3"
ENV DB_NAME="ehabo-PolybotService-DynamoDB"

CMD ["python3", "app.py"]
