# Xây dựng hệ thống huấn luyện mô hình bằng Kubeflow

Sau khi gán nhãn dữ liệu, huấn luyện mô hình và triển khai mô hình trên giao diện web. Chúng ta có thể nhận thấy việc thực hiện công việc này đang độc lập với nhau và thực hiện một cách thủ công.

Trong phần này, chúng ta sẽ sử dụng Kubeflow để xây dựng một hệ thống để kết nối các bước tiền xử lý dữ liệu, huấn luyện mô hình, triển khai mô hình một cách tự động mỗi khi chúng ta có dữ liệu mới, thay vì phải thực hiện một cách thủ công như chúng ta đã thực hiện ở các bài trước.

Với Kubeflow, chúng ta có thể triển khai một hệ thống học máy trên Kubernetes một cách dễ dàng và có thể nâng cấp và mở rộng tuỳ theo nhu cầu. Các bạn có thể tìm hiểu thêm về Kubeflow ở đây: https://www.kubeflow.org/docs/started/kubeflow-overview/

## Cài đặt Kubeflow
Chúng ta có thể cài đặt Kubeflow trên nhiều môi trường khác nhau, có thể trên máy tính cá nhân, `on-premise` hoặc trên các nền tảng đám mây như `Amazon Web Services (AWS)`, `Microsoft Azure`, `Google Cloud`, etc. 

### Cài đặt Kubeflow bằng MiniKF
Trong phần này, mình sẽ cài đặt Kubeflow trên máy tính cá nhân với [Arrikato MiniKF](https://www.kubeflow.org/docs/distributions/minikf/). Các bạn có thể làm theo hướng dẫn trong đường dẫn này hoặc làm theo các bước dưới đây.

Để cài đặt `MiniKF`, trước hết chúng ta cần cài đặt `Vagrant` và `Virtual Box`. Ở đây mình sử dụng hệ điều hành Ubuntu.

Cài đặt `Vagrant`:
```
curl -fsSL https://apt.releases.hashicorp.com/gpg | sudo apt-key add -
sudo apt-add-repository "deb [arch=amd64] https://apt.releases.hashicorp.com $(lsb_release -cs) main"
sudo apt-get update && sudo apt-get install vagrant
```

Với `Virtual Box`, các bạn tải xuống phiên bản tương ứng với hệ điều hành của mình và tiến hành cài đặt như thông thường theo hướng dẫn ở đây: https://www.virtualbox.org/wiki/Linux_Downloads

Sau khi cài đặt hoàn tất, chúng ta sẽ sử dụng `Vagrant` để cài đặt MiniKF và khởi động MiniKF:

```
vagrant init arrikto/minikf
vagrant up
```

Sau khi cài đặt và khởi động hoàn tất, chúng ta truy cập vào địa chỉ [http://10.10.10.10](http://10.10.10.10) để tiếp tục cài đặt và bắt đàu sử dụng Kubeflow. Giao diện cài đặt trên địa chỉ web như hình dưới.

![alt text](./images/kubeflow.png "Kubeflow")

Chúng ta cài đặt theo giao diện trên `Terminal`, sau khi cài đặt, nhấn `Start MiniKF` và sử dụng tên tài khoản và mật khẩu trên giao diện để đăng nhập vào Kubeflow.

### Cài đặt Kubeflow bằng trên Microk8s
Các bạn có thể làm theo hướng dẫn ở đường dẫn này hoặc các bước sau đây:
1. Cài đặt Microk8s
```
sudo snap install microk8s --classic
```

2. Thêm người dùng hiện tại vào `admin group`
```
sudo usermod -a -G microk8s $USER
sudo chown -f -R $USER ~/.kube
```

sau đó đăng xuất và đăng nhập lại.
3. Kiểm tra xem Microk8s để đảm bảo Microk8s đang chạy
```
microk8s status --wait-ready
```
4. Cài đặt Kubeflow
```
microk8s enable kubeflow
```

Để sử dụng GPU, chúng ta sử dụng lệnh sau:
```
microk8s enable gpu
```

Sau khi cài đặt thành công, chúng ta truy cập vào địa chỉ: 10.64.140.43.nip.io để sử dụng Kubeflow

Đây là giao diện của Kubeflow sau khi đăng nhập:

Kubeflow UI có một số tính năng như:
- `Home`: giao diện chính quản lý các thành phần của Kubeflow
- `Pipelines` dành cho việc quản lý các `pipeline`
- `Notebook Servers` để sử dụng Jupyter notebooks.
- `Katib` cho việc tối ưu siêu tham số.
- `Artifact Store` để quản lý `artifact`.
- `Manage Contributors` để chia sẻ quyền truy cập của người dùng trên các `namespace` trong Kubeflow.

## Xây dựng Kubeflow Pipeline
Trong phần này, chúng ta sẽ xây dựng một Kubeflow pipeline bao gốm các thành phần: tiền xử lý dữ liệu, huấn luyện mô hình, và triển khai mô hình.

### Cấu trúc thư mục 
Để dễ dàng xây dựng các thành phần của Kubeflow, chúng ta sẽ refactor lại code mà chúng ta đã sử dụng ở bước trên trong jupyter notebook.

Cấu trúc của một thư mục để xây dựng Kubeflow pipeline như sau:
```
components
├── preprocess
│   ├── src
│   │  └── main.py
│   ├── Dockerfile
│   ├── build_image.sh
│   └── component.yaml
├── train
│   └── ...
└── _deployment
    └── ...
```

Chúng ta sẽ chia thành ba thành phần nhỏ: `preprocess`, `train`, `deployment`. Trong mỗi thành phần, chúng ta có các thành phần con như sau:
- **Thư mục `src`**: dùng để chứa code cho nhiệm vụ mà chúng ta cần làm: tiền xử lý dữ liệu, huấn luyện mô hình hay triển khai mô hình. Chúng ta sẽ refactor và chuyển code chúng ta đã làm ở các bài trước vào thư mục này.
- `Dockerfile`: dùng để xây dựng `docker image` làm môi trường cho chương trình.
- `build_image.sh`: shell dùng để khởi tạo `docker image` và đẩy chúng lên `docker hub`
- `component.yaml`: định nghĩa nội dung của thành phần

Chúng ta sẽ đi vào chi tiết của từng thành phần trong phần tiếp theo. Chúng ta sẽ xây dựng một `pipeline` như sau:

![alt text](./images/pipeline.png "Pipeline")

### Tiền xử lý dữ liệu
Chúng ta sẽ cùng nhau tiến hành xây dựng thành phần đầu tiên đó là tiền xử lý dữ liệu. 

#### Xây dựng mã nguồn
Đầu tiên, chúng ta cần sao chép mã nguồn của thành phần tiền xử lý dữ liệu ở các bài trước và đưa vào tệp `main.py` trong thư mục `src`. Tệp `main.py` có nội dung như sau:

```python
import os
import json
from pathlib import Path

# define the aws key
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
    
# split train/val/test/user_data label from this json file
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

# write data path to output_path.txt
data_path = 's3://ocrpipeline/data'
Path('/output_path.txt').write_text(data_path)
```

Các bước thực hiện như sau:

0. Khai báo các `aws keys` để có thể tải lên và tải xuống dữ liệu từ `AWS S3`: Trong thực tế, chúng ta không cần khai báo các biến này ở trong mã nguồn của mình, chúng ta sẽ thay thế nó bằng cách cho chúng là biến môi trường trước khi chạy `docker image`. Chúng ta không nên khai báo các `aws keys` trong code vì người khác có thể sử dụng được tài khoản của mình và sẽ ảnh hưởng đến vấn đề bảo mật.
1. Tải dữ liệu đã được gán nhán từ `label studio`: Do chúng ta đang sử dụng Label Studio ở máy cục bộ (local host), vì vậy, để gửi được `request` từ `container` trong Kubeflow, chúng ta cần tìm cách mở (expose) Label Studio API. Một cách đơn giản là chúng ta có thể sử dụng `ngrok`. Để sử dụng `ngrok`, các bạn vào đường dẫn này và làm theo các bước để tải và cài đặt `ngrok`. Do `Label Studio` đang chạy ở cổng 8080, vì vậy chúng ta expose `Label Studio` bằng `ngrok` như sau: 
    ```
    ./ngrok http 8080
    ```

    Chúng ta sẽ có đường đường dẫn mà `ngrok` sinh ra và sử dụng để tải dữ liệu từ Label Studio.

2. Xử lý dữ liệu `label studio` và chuyển về định dạng mới
3. Tải dữ liệu lên `AWS S3`
4. Ghi đường dẫn dữ liệu trên `AWS S3` vào tệp `output_path.txt` để có thể sử dụng trong thành phần kế tiếp.

#### Xây dựng Docker image
Chúng ta sẽ xây dựng Docker image để làm môi trường chạy mã nguồn ở trên như sau:
```dockerfile
FROM python:3.7
COPY ./src /src
```

Ở đây chúng ta sử dụng môi trường `python 3.7` và tiến hành sao chép mã nguồn trong thư mục `src`vào thư mục `/src` của `docker container`.

#### Xây dựng shell build_image.sh
Chúng ta xây dựng shell với nội dung như sau:
```shell
#!/bin/bash

full_image_name=<username>/ocrpreprocess

docker build -t "${full_image_name}" .
docker push "$full_image_name"
```

Chúng ta đặt tên cho `docker image` là `<username>/ocrpreprocess` (bạn có thể thay `<username>` với tên của bạn) và tiến hành xây dựng nó với câu lệnh `docker build`. Sau đó chúng ta sẽ đẩy `docker image` này lên `docker hub`.

Để có thể đẩy `docker image` lên `docker hub`, chúng ta cần có tài khoản tại https://hub.docker.com và đăng nhập docker bằng câu lệnh 
```
docker login
```

Sau khi đăng nhập `docker hub`, chúng ta tiến hành chạy shell để xây dựng docker image như sau:

```
chmod +x build_image.sh
./build_image.sh
```

Như vậy chúng ta đã xây dựng môi trường để thực hiện việc tiền xử lý dữ liệu.

### Huấn luyện mô hình

### Triển khai mô hình








Sau khi deploy mô hình, chúng ta thực hiện port forwarding và send request như sau:
```
kubectl port-forward --namespace kubeflow-user  $(kubectl get pod --namespace kubeflow-user --output jsonpath='{.items[0].metadata.name}') 8080:5000
```

```
curl -i -v -H "Host: http://custom-simple.kubeflow-user.example.com" -X POST "http://localhost:8080/predict" -F file=@Downloads/plane.jpg
```