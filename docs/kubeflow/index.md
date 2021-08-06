# Xây dựng hệ thống huấn luyện mô hình bằng Kubeflow

Sau khi gán nhãn dữ liệu, huấn luyện mô hình và triển khai mô hình trên giao diện web. Chúng ta có thể nhận thấy việc thực hiện công việc này đang độc lập với nhau và thực hiện một cách thủ công.

Trong phần này, chúng ta sẽ sử dụng Kubeflow để xây dựng một hệ thống để kết nối các bước tiền xử lý dữ liệu, huấn luyện mô hình, triển khai mô hình một cách tự động mỗi khi chúng ta có dữ liệu mới, thay vì phải thực hiện một cách thủ công như chúng ta đã thực hiện ở các bài trước.

Với Kubeflow, chúng ta có thể triển khai một hệ thống học máy trên Kubernetes một cách dễ dàng và có thể nâng cấp và mở rộng tuỳ theo nhu cầu. Các bạn có thể tìm hiểu thêm về Kubeflow ở đây: https://www.kubeflow.org/docs/started/kubeflow-overview/

## Cài đặt Kubeflow
Chúng ta có thể cài đặt Kubeflow trên nhiều môi trường khác nhau, có thể trên máy tính cá nhân, `on-premise` hoặc trên các nền tảng đám mây như `Amazon Web Services (AWS)`, `Microsoft Azure`, `Google Cloud`, etc. 

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

Chúng ta cài đặt theo giao diện trên `Terminal`, sau khi cài đặt, nhấn `Start MiniKF` và sử dụng tên tài khoản và mật khẩu trên giao diện để đăng nhập vào Kubeflow.

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


