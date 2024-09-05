import json
import time
import uuid
from decimal import Decimal
from pathlib import Path
import requests
from detect import run
import yaml
from loguru import logger
import os
import boto3
from urllib.parse import urljoin

images_bucket = os.environ['BUCKET_NAME']
queue_name = os.environ['SQS_QUEUE_NAME']
region_name = os.environ['REGION_NAME']
TELEGRAM_APP_URL = os.environ['TELEGRAM_APP_URL']
logger.info(f'TELEGRAM_APP_URL: {TELEGRAM_APP_URL}.')
if TELEGRAM_APP_URL is None or TELEGRAM_APP_URL.strip() == "":
    raise ValueError("TELEGRAM_APP_URL environment variable is not set or is empty.")

sqs_client = boto3.client('sqs', region_name=region_name)

with open("data/coco128.yaml", "r") as stream:
    names = yaml.safe_load(stream)['names']


def consume():
    # The function runs in an infinite loop, continually polling the SQS queue for new messages.
    while True:
        # Receive Message from SQS
        response = sqs_client.receive_message(QueueUrl=queue_name, MaxNumberOfMessages=1, WaitTimeSeconds=5)
        # Check for Messages:
        if 'Messages' in response:
            # Extract message details
            message_body = response['Messages'][0]['Body']
            receipt_handle = response['Messages'][0]['ReceiptHandle']
            # Parses the message body from JSON format to a Python dictionary and retrieves the message ID
            message = json.loads(message_body)
            prediction_id = response['Messages'][0]['MessageId']
            logger.info(f'Prediction: {prediction_id}. Start processing')
            # Retrieve Chat ID and Image Name
            chat_id = message.get('chat_id')
            img_name = message.get('imgName')
            if not img_name or not chat_id:
                logger.error('Invalid message format: chat_id or imgName missing')
                sqs_client.delete_message(QueueUrl=queue_name, ReceiptHandle=receipt_handle)
                continue

            logger.info(f'img_name received: {img_name}')
            photo_s3_name = img_name.split("/")
            file_path_pic_download = os.path.join(os.getcwd(), photo_s3_name[1])
            logger.info(f'Download path: {file_path_pic_download}')
            # Download Image from S3
            s3_client = boto3.client('s3')
            s3_client.download_file(images_bucket, photo_s3_name[1], file_path_pic_download)

            original_img_path = file_path_pic_download
            logger.info(f'Prediction: {prediction_id}{original_img_path}. Download img completed')
            # Predicts the objects in the image
            run(
                weights='yolov5s.pt',
                data='data/coco128.yaml',
                source=original_img_path,
                project='static/data',
                name=prediction_id,
                save_txt=True
            )

            logger.info(f'prediction: {prediction_id}/{original_img_path}. done')

            predicted_img_path = Path(f'static/data/{prediction_id}/{str(photo_s3_name[1])}')
            logger.info(f'predicted_img_path: {predicted_img_path}.')
            # Upload predicted image to S3
            unique_filename = str(uuid.uuid4()) + '.jpeg'
            s3_client.upload_file(str(predicted_img_path), images_bucket, unique_filename)
            logger.info("upload to s3.")
            # Parse prediction labels and create a summary
            pred_summary_path = Path(f'static/data/{prediction_id}/labels/{photo_s3_name[1].split(".")[0]}.txt')
            logger.info(f'pred_summary_path: {pred_summary_path}.')
            if pred_summary_path.exists():
                with open(pred_summary_path) as f:
                    labels = f.read().splitlines()
                    labels = [line.split(' ') for line in labels]
                    labels = [{
                        'class': names[int(l[0])],
                        'cx': Decimal(l[1]),
                        'cy': Decimal(l[2]),
                        'width': Decimal(l[3]),
                        'height': Decimal(l[4]),
                    } for l in labels]

                logger.info(f'prediction: {prediction_id}/{original_img_path}. prediction summary:\n\n{labels}')
                chat_id = str(chat_id)  # Convert chat_id to string
                prediction_summary = {
                    'prediction_id': prediction_id,
                    'chat_id': chat_id,
                    'original_img_path': original_img_path,
                    'predicted_img_path': str(predicted_img_path),
                    'labels': labels,
                    'unique_filename': unique_filename,
                    'time': Decimal(time.time())
                }
                # TODO store the prediction_summary in a DynamoDB table
                # TODO perform a GET request to Polybot to `/results` endpoint
                # Store the prediction_summary in a DynamoDB table

                dynamodb = boto3.resource('dynamodb', region_name=region_name)
                logger.info({dynamodb})
                table = dynamodb.Table('ehabo-PolybotService-DynamoDB-tf')
                logger.info({table})
                table.put_item(Item=prediction_summary)
                full_url = urljoin(TELEGRAM_APP_URL, 'results')
                # Send the message from my yolo5 to load balancer:
                try:
                    response = requests.post(f'{full_url}', params={'predictionId': prediction_id})
                    response.raise_for_status()  # Raise an error for bad status codes
                    logger.info(f'prediction: {prediction_id}. Notified Polybot microservice successfully')
                except requests.exceptions.RequestException as e:
                    logger.error(f'prediction: {prediction_id}. Failed to notify Polybot microservice. Error: {str(e)}')
                    if e.response is not None:
                        logger.error(f'Response status code: {response.status_code}')
                        logger.error(f'Response text: {response.text}')

            else:
                logger.error(f'Prediction: {prediction_id}{original_img_path}. prediction result not found')
                sqs_client.delete_message(QueueUrl=queue_name, ReceiptHandle=receipt_handle)
                continue

            # Delete the message from the queue as the job is considered as DONE
            sqs_client.delete_message(QueueUrl=queue_name, ReceiptHandle=receipt_handle)


if __name__ == "__main__":
    consume()
