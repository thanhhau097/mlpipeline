from botocore.retries import bucket


def deploy_model(model_s3_path: str):
    char_ords = [97, 225, 224, 7841, 7843, 227, 259, 7855, 7857, 7859, 7861, 7863, 226, 7845, 7847, 7849, 7851, 7853, 98, 99, 100, 273, 101, 233, 232, 7865, 7867, 7869, 234, 7871, 7873, 7879, 7875, 7877, 102, 103, 104, 105, 237, 236, 7883, 7881, 297, 106, 107, 108, 109, 110, 111, 243, 242, 7885, 7887, 245, 244, 7889, 7891, 7897, 7893, 7895, 417, 7899, 7901, 7907, 7903, 7905, 112, 113, 114, 115, 116, 117, 250, 249, 7909, 7911, 361, 432, 7913, 7915, 7921, 7917, 7919, 118, 120, 121, 253, 7923, 7927, 7929, 7925, 122, 119, 65, 193, 192, 7840, 7842, 195, 258, 7854, 7856, 7858, 7860, 7862, 194, 7844, 7846, 7848, 7850, 7852, 66, 67, 68, 272, 69, 201, 200, 7864, 7866, 7868, 202, 7870, 7872, 7878, 7874, 7876, 70, 71, 72, 73, 205, 204, 7882, 7880, 296, 74, 75, 76, 77, 78, 79, 211, 210, 7884, 7886, 213, 212, 7888, 7890, 7896, 7892, 7894, 416, 7898, 7900, 7906, 7902, 7904, 80, 81, 82, 83, 84, 85, 218, 217, 7908, 7910, 360, 431, 7912, 7914, 7920, 7916, 7918, 86, 88, 89, 221, 7922, 7924, 7926, 7928, 90, 87, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 32, 46, 44, 45, 47, 40, 41, 39, 35, 43, 58]
    CHARACTERS = [chr(c) for c in char_ords]
    PAD_token = 0  # Used for padding short sentences
    SOS_token = 1  # Start-of-sentence token
    EOS_token = 2  # End-of-sentence token

    CHAR2INDEX = {"PAD": PAD_token, "SOS": SOS_token, "EOS": EOS_token}
    INDEX2CHAR = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}

    for i, c in enumerate(CHARACTERS):
        CHAR2INDEX[c] = i + 3
        INDEX2CHAR[i + 3] = c
    
    def get_indices_from_label(label):
        indices = []
        for char in label:
    #         if CHAR2INDEX.get(char) is not None:
                indices.append(CHAR2INDEX[char])

        indices.append(EOS_token)
        return indices

    def get_label_from_indices(indices):
        label = ""
        for index in indices:
            if index == EOS_token:
                break
            elif index == PAD_token:
                continue
            else:
                label += INDEX2CHAR[index.item()]

        return label
    
    import torch.nn as nn
    import torch.nn.functional as F


    class VGG_FeatureExtractor(nn.Module):
        """ FeatureExtractor of CRNN (https://arxiv.org/pdf/1507.05717.pdf) """

        def __init__(self, input_channel, output_channel=512):
            super(VGG_FeatureExtractor, self).__init__()
            self.output_channel = [int(output_channel / 8), int(output_channel / 4),
                                   int(output_channel / 2), output_channel]  # [64, 128, 256, 512]
            self.ConvNet = nn.Sequential(
                nn.Conv2d(input_channel, self.output_channel[0], 3, 1, 1), nn.ReLU(True),
                nn.MaxPool2d(2, 2),  # 64x16x50
                nn.Conv2d(self.output_channel[0], self.output_channel[1], 3, 1, 1), nn.ReLU(True),
                nn.MaxPool2d(2, 2),  # 128x8x25
                nn.Conv2d(self.output_channel[1], self.output_channel[2], 3, 1, 1), nn.ReLU(True),  # 256x8x25
                nn.Conv2d(self.output_channel[2], self.output_channel[2], 3, 1, 1), nn.ReLU(True),
                nn.MaxPool2d((2, 1), (2, 1)),  # 256x4x25
                nn.Conv2d(self.output_channel[2], self.output_channel[3], 3, 1, 1, bias=False),
                nn.BatchNorm2d(self.output_channel[3]), nn.ReLU(True),  # 512x4x25
                nn.Conv2d(self.output_channel[3], self.output_channel[3], 3, 1, 1, bias=False),
                nn.BatchNorm2d(self.output_channel[3]), nn.ReLU(True),
                nn.MaxPool2d((2, 1), (2, 1)),  # 512x2x25
                nn.Conv2d(self.output_channel[3], self.output_channel[3], 2, 1, 0), nn.ReLU(True))  # 512x1x24

        def forward(self, input):
            return self.ConvNet(input)
        
    
    class BidirectionalGRU(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(BidirectionalGRU, self).__init__()

            self.rnn = nn.GRU(input_size, hidden_size, bidirectional=True)
            self.embedding = nn.Linear(hidden_size * 2, output_size)

        def forward(self, x):
            recurrent, hidden = self.rnn(x)
            T, b, h = recurrent.size()
            t_rec = recurrent.view(T * b, h)

            output = self.embedding(t_rec)  # [T * b, nOut]
            output = output.view(T, b, -1)

            return output, hidden
    
    class CTCModel(nn.Module):
        def __init__(self, inner_dim=512, num_chars=65):
            super().__init__()
            self.encoder = VGG_FeatureExtractor(3, inner_dim)
            self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))
            self.rnn_encoder = BidirectionalGRU(inner_dim, 256, 256)
            self.num_chars = num_chars
            self.decoder = nn.Linear(256, self.num_chars)

        def forward(self, x, labels=None, max_label_length=None, device=None, training=True):
            # ---------------- CNN ENCODER --------------
            x = self.encoder(x)
            # print('After CNN:', x.size())

            # ---------------- CNN TO RNN ----------------
            x = x.permute(3, 0, 1, 2)  # from B x C x H x W -> W x B x C x H
            x = self.AdaptiveAvgPool(x)
            size = x.size()
            x = x.reshape(size[0], size[1], size[2] * size[3])

            # ----------------- RNN ENCODER ---------------
            encoder_outputs, last_hidden = self.rnn_encoder(x)
            # print('After RNN', x.size())

            # --------------- CTC DECODER -------------------
            # batch_size = encoder_outputs.size()[1]
            outputs = self.decoder(encoder_outputs)

            return outputs
        
    import uuid

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

    import boto3
    import torch

    import os
    os.environ["AWS_ACCESS_KEY_ID"] = "*********************"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "*********************"

    s3 = boto3.client('s3')
    bucket_name = model_s3_path.split('s3://')[1].split('/')[0]
    object_name = model_s3_path.split('s3://' + bucket_name)[1][1:]
    model_path = 'best_model.pth'
    s3.download_file(bucket_name, object_name, model_path)

    model = CTCModel(inner_dim=128, num_chars=len(CHAR2INDEX))
    device = 'cpu' if not torch.cuda.is_available() else 'cuda'
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)

    def predict(image):
        # upload image to aws s3
        filename = str(uuid.uuid4()) + '.png'
        cv2.imwrite(filename, image)
        upload_file_to_s3(filename, bucket='ocrpipeline', object_name='data/user_data/images/')

        model.eval()
        batch = [{'image': image, 'label': [1]}]
        images = collate_wrapper(batch)[0]

        images = images.to(device)

        outputs = model(images)
        outputs = outputs.permute(1, 0, 2)
        output = outputs[0]

        out_best = list(torch.argmax(output, -1))  # [2:]
        out_best = [k for k, g in itertools.groupby(out_best)]
        pred_text = get_label_from_indices(out_best)

        return pred_text
        

    # import gradio
    # device = 'cpu' if not torch.cuda.is_available() else 'cuda'
    # #     model.load_state_dict(torch.load('best_model.pth'))
    # model = model.to(device)
    # interface = gradio.Interface(predict, "image", "text")
    
    # interface.launch()
    
    import os
    from flask import Flask, flash, request, redirect, url_for
    from werkzeug.utils import secure_filename

    UPLOAD_FOLDER = './data/user_data/'
    ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

    app = Flask(__name__)
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.secret_key = "super secret key"


    @app.route("/predict", methods=['POST'])
    def upload_file():
        if request.method == 'POST':
            # check if the post request has the file part
            if 'file' not in request.files:
                flash('No file part')
                return redirect(request.url)
            file = request.files['file']
            # if user does not select file, browser also
            # submit a empty part without filename
            if file.filename == '':
                flash('No selected file')
                return redirect(request.url)
            if file:
                print('found file')
                filename = secure_filename(file.filename)
                filepath = os.path.join(filename)
                file.save(filepath)
                
                image = cv2.imread(filepath)
                output = predict(image)
                return output


    app.run(debug=False, threaded=False, host='0.0.0.0', port=5000)
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Deployment')
    parser.add_argument('--model_s3_path', type=str)

    args = parser.parse_args()

    deploy_model(model_s3_path=args.model_s3_path)

