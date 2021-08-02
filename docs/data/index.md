## Data Pipeline
Trong phần này, chúng ta sẽ thực hiện việc lưu trữ và gán nhãn dữ liệu.

### 1. Phân tích dữ liệu
Trước khi giải quyết một bài toán, chúng ta cần phải thực hiện công việc phân tích dữ liệu để đưa ra phương pháp gán nhãn phù hợp và hiệu quả. Bài toán mà chúng ta sẽ cùng nhau giải quyết trong blog này tương đối đơn giản về mặt dữ liệu, với đầu vào là hình ảnh của một dòng chữ viết tay và chúng ta cần trả về kết quả dãy kí tự mà máy đọc được từ hình ảnh đó.

```python
import os
import cv2
from matplotlib import pyplot as plt
```

### Data Sample


```python
DATA_SAMPLE_FOLDER = './Challenge 1_ Handwriting OCR for Vietnamese Address/0825_DataSamples 1'
```


```python
for image_name in os.listdir(DATA_SAMPLE_FOLDER)[:6]:
    if '.json' in image_name:
        continue
    
    image_path = os.path.join(DATA_SAMPLE_FOLDER, image_name)
    image = cv2.imread(image_path)
    plt.imshow(image)
    plt.show()
```


    
![png](./images/output_3_0.png)
    



    
![png](./images/output_3_1.png)
    



    
![png](./images/output_3_2.png)
    



    
![png](./images/output_3_3.png)
    



    
![png](./images/output_3_4.png)
    


### Dataset
Thông thường, khi tiếp cận các bài toán học máy, chúng ta cần phân tích dữ liệu và chia dữ liệu thành các tập huấn luyện (training set), tập đánh giá (validation set) và tập thử nghiệm (testing set). 

Trong bộ dữ liệu mà chúng ta sử dụng ở đây, tập dữ liệu huấn luyện và tập thử nghiệm đã được chia sẵn, vì vậy, chúng ta sẽ chia tập dữ liệu huấn luyện có sẵn thành hai tập: tập dữ liệu huấn luyện và tập đánh giá.


```python
TRAIN_FOLDER = './Challenge 1_ Handwriting OCR for Vietnamese Address/0916_Data Samples 2'
TEST_FOLDER = './Challenge 1_ Handwriting OCR for Vietnamese Address/1015_Private Test'
```


```python
train_image_paths = [os.path.join(TRAIN_FOLDER, image_name) 
                     for image_name in os.listdir(TRAIN_FOLDER) 
                     if '.json' not in image_name]
train_image_paths[:5]
```


    ['./Challenge 1_ Handwriting OCR for Vietnamese Address/0916_Data Samples 2/0768_samples.png',
     './Challenge 1_ Handwriting OCR for Vietnamese Address/0916_Data Samples 2/0238_samples.png',
     './Challenge 1_ Handwriting OCR for Vietnamese Address/0916_Data Samples 2/0898_samples.png',
     './Challenge 1_ Handwriting OCR for Vietnamese Address/0916_Data Samples 2/0907_samples.png',
     './Challenge 1_ Handwriting OCR for Vietnamese Address/0916_Data Samples 2/0071_samples.png']



```python
print('Number of training images:', len(train_image_paths))
```

    Number of training images: 1823



```python
test_image_paths = [os.path.join(TEST_FOLDER, image_name) 
                     for image_name in os.listdir(TEST_FOLDER) 
                     if '.json' not in image_name]
test_image_paths[:5]
```




    ['./Challenge 1_ Handwriting OCR for Vietnamese Address/1015_Private Test/0232_tests.png',
     './Challenge 1_ Handwriting OCR for Vietnamese Address/1015_Private Test/0004_tests.png',
     './Challenge 1_ Handwriting OCR for Vietnamese Address/1015_Private Test/0374_tests.png',
     './Challenge 1_ Handwriting OCR for Vietnamese Address/1015_Private Test/0142_tests.png',
     './Challenge 1_ Handwriting OCR for Vietnamese Address/1015_Private Test/0468_tests.png']




```python
print('Number of testing images:', len(test_image_paths))
```

    Number of testing images: 549


#### Split datasets


```python
NEW_TRAIN_FOLDER = './train/images'
NEW_VALIDATION_FOLDER = './validation/images'
NEW_TEST_FOLDER = './test/images'

for folder in [NEW_TRAIN_FOLDER, NEW_VALIDATION_FOLDER, NEW_TEST_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
```


```python
import random

validation_image_paths = random.choices(train_image_paths, k=int(0.2*len(train_image_paths)))
train_image_paths = list(set(train_image_paths).difference(set(validation_image_paths)))
```


```python
print('Number of training images:', len(train_image_paths))
print('Number of validation images:', len(validation_image_paths))
print('Number of testing images:', len(test_image_paths))
```

    Number of training images: 1497
    Number of validation images: 364
    Number of testing images: 549



```python
import shutil

def copy_images_to_folder(image_paths, to_folder):
    for path in image_paths:
        shutil.copy2(path, to_folder)
        
copy_images_to_folder(train_image_paths, NEW_TRAIN_FOLDER)
copy_images_to_folder(validation_image_paths, NEW_VALIDATION_FOLDER)
copy_images_to_folder(test_image_paths, NEW_TEST_FOLDER)
```

