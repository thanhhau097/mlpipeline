# Huấn luyện mô hình
Sau khi có được dữ liệu thông qua việc gán nhãn, chúng ta sẽ tiến hành xây dựng và huấn luyện mô hình học máy dựa trên dữ liệu này. Trong blog này, chúng ta sẽ sử dụng Pytorch để xây dựng và huấn luyện mô hình, Cometml để quản lý thí nghiệm và tối ưu siêu tham số và DVC để lưu trọng số của mô hình.

## Bài toán
Mục tiêu của bài toán là đọc hình ảnh của một dòng chữ và đưa ra kết quả dòng chữ đó. Ví dụ:

- Đầu vào: 

    ![png](../data/images/output_3_0.png)

- Kết quả: `phòng 101, tầng 1, lô 04 - TT58, khu đô thị Tây Nam Linh Đàm`

Chúng ta sẽ gọi bài toán này là bài toán `Nhận diện kí tự quang học (Optical Character Recognition - OCR)`
## Các phương pháp/mô hình phổ biến
Có rất nhiều phương pháp, mô hình học máy có thể được sử dụng để giải quyết bài toán OCR, các bạn có thể tìm hiểu thêm ở github này: https://github.com/hwalsuklee/awesome-deep-text-detection-recognition

Trong phần này, chúng ta sẽ nhắc đến hai mô hình phổ biến là nền tảng của các mô hình sau này, đó là `Convolutional Recurrent Neural Network + CTC` và `Convolution Recurrent Neural Network + Attention`. Có nhiều blog/video giải thích các mô hình này một cách rất chi tiết và dễ hiểu, đường dẫn đến các blog này được đặt ở mục tài liệu tham khảo. Ở trong blog này, chúng ta sẽ tập trung vào phần thực hành để hiểu rõ hơn về cơ chế hoạt động và cách cải thiện mô hình.

![alt text](./images/crnn.png "CRNN")

Trong blog này, chúng ta sẽ sử dụng `CTC decoding` cho `decoding algorithm` và `CTC loss function` để huấn luyện mô hình.

## Xây dựng mô hình học máy với Convolutional Recurrent Neural Network

## Quản lý thí nghiệm

## Tối ưu siêu tham số

## Model Versioning

## Tổng kết

## Tài liệu tham khảo
1. [An Intuitive Explanation of Connectionist Temporal Classification](https://towardsdatascience.com/intuitively-understanding-connectionist-temporal-classification-3797e43a86c)
2. [How to build end-to-end recognition system](https://www.youtube.com/watch?v=uVbOckyUemo)