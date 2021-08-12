import os
import json
from pathlib import Path


os.environ["AWS_ACCESS_KEY_ID"] = "**************************"
os.environ["AWS_SECRET_ACCESS_KEY"] = "**************************"

# download data from label studio
print("DOWNLOADING DATA")
import requests

url = "https://***********.ngrok.io/api/projects/1/export"
querystring = {"exportType":"JSON"}
headers = {
    'authorization': "Token ******************************",
}
response = requests.request("GET", url, headers=headers, params=querystring)
    
# TODO: split train/val/test/user_data label from this json file
user_data = []
train_data = []
val_data = []
test_data = []
for item in response.json():
    if 's3://ocrpipeline/data/user_data' in item['data']['captioning']:
        user_data.append(item)
    elif 's3://ocrpipeline/data/train' in item['data']['captioning']:
        train_data.append(item)
    elif 's3://ocrpipeline/data/test' in item['data']['captioning']:
        test_data.append(item)
    elif 's3://ocrpipeline/data/validation' in item['data']['captioning']:
        val_data.append(item)

def save_labels(folder, data):
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    label_path = os.path.join(folder, 'label_studio_data.json')
    with open(label_path, 'w') as f:
        json.dump(data, f)

save_labels('./data/train/', train_data)
save_labels('./data/test/', test_data)
save_labels('./data/validation/', val_data)
save_labels('./data/user_data/', user_data)

# preprocess
print("PREPROCESSING DATA")
def convert_label_studio_format_to_ocr_format(label_studio_json_path, output_path):
    with open(label_studio_json_path, 'r') as f:
        data = json.load(f)

    ocr_data = {}

    for item in data:
        image_name = os.path.basename(item['data']['captioning'])

        text = ''
        for value_item in item['annotations'][0]['result']:
            if value_item['from_name'] == 'caption':
                text = value_item['value']['text'][0]
        ocr_data[image_name] = text

    with open(output_path, 'w') as f:
        json.dump(ocr_data, f, indent=4)

    print('Successfully converted ', label_studio_json_path)
convert_label_studio_format_to_ocr_format('./data/train/label_studio_data.json', './data/train/labels.json')
convert_label_studio_format_to_ocr_format('./data/validation/label_studio_data.json', './data/validation/labels.json')
convert_label_studio_format_to_ocr_format('./data/test/label_studio_data.json', './data/test/labels.json')
convert_label_studio_format_to_ocr_format('./data/user_data/label_studio_data.json', './data/user_data/labels.json')

# upload these file to s3
print("UPLOADING DATA")
import logging
import boto3
from botocore.exceptions import ClientError


def upload_file_to_s3(file_name, bucket, object_name=None):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = file_name

    # Upload the file
    s3_client = boto3.client('s3')
    try:
        response = s3_client.upload_file(file_name, bucket, object_name)
    except ClientError as e:
        logging.error(e)
        return False
    return True


upload_file_to_s3('data/train/labels.json', bucket='ocrpipeline', object_name='data/train/labels.json')
upload_file_to_s3('data/validation/labels.json', bucket='ocrpipeline', object_name='data/validation/labels.json')
upload_file_to_s3('data/test/labels.json', bucket='ocrpipeline', object_name='data/test/labels.json')
upload_file_to_s3('data/user_data/labels.json', bucket='ocrpipeline', object_name='data/user_data/labels.json')

data_path = 's3://ocrpipeline/data'
Path('/output_path.txt').write_text(data_path)