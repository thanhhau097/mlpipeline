def train_model(data_path):
    import os
    from pathlib import Path
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


    os.environ["AWS_ACCESS_KEY_ID"] = "AKIAR3JE4MXNRIY3B4EI"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "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"

    CHARACTERS = "aáàạảãăắằẳẵặâấầẩẫậbcdđeéèẹẻẽêếềệểễfghiíìịỉĩjklmnoóòọỏõôốồộổỗơớờợởỡpqrstuúùụủũưứừựửữvxyýỳỷỹỵzwAÁÀẠẢÃĂẮẰẲẴẶÂẤẦẨẪẬBCDĐEÉÈẸẺẼÊẾỀỆỂỄFGHIÍÌỊỈĨJKLMNOÓÒỌỎÕÔỐỒỘỔỖƠỚỜỢỞỠPQRSTUÚÙỤỦŨƯỨỪỰỬỮVXYÝỲỴỶỸZW0123456789 .,-/()'#+:"
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

    from comet_ml import Experiment

    import cv2
    from torch.utils.data import Dataset


    class OCRDataset(Dataset):
        def __init__(self, data_dir, label_path, transform=None):
            self.data_dir = data_dir
            self.label_path = label_path
            self.transform = transform
            self.image_paths, self.labels = self.get_image_paths_and_labels(label_path)

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            img_path = self.get_data_path(self.image_paths[idx])
            image = cv2.imread(img_path)
            label = self.labels[idx]
            label = get_indices_from_label(label)
            sample = {"image": image, "label": label}
            return sample

        def get_data_path(self, path):
            return os.path.join(self.data_dir, path)

        def get_image_paths_and_labels(self, json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            image_paths = list(data.keys())
            labels = list(data.values())
            return image_paths, labels
        
        
    import itertools

    def collate_wrapper(batch):
        """
        Labels are already numbers
        :param batch:
        :return:
        """
        images = []
        labels = []
        # TODO: can change height in config
        height = 64
        max_width = 0
        max_label_length = 0

        for sample in batch:
            image = sample['image']
            try:
                image = process_image(image, height=height, channels=image.shape[2])
            except:
                continue

            if image.shape[1] > max_width:
                max_width = image.shape[1]

            label = sample['label']

            if len(label) > max_label_length:
                max_label_length = len(label)

            images.append(image)
            labels.append(label)

        # PAD IMAGES: convert to tensor with size b x c x h x w (from b x h x w x c)
        channels = images[0].shape[2]
        images = process_batch_images(images, height=height, max_width=max_width, channels=channels)
        images = images.transpose((0, 3, 1, 2))
        images = torch.from_numpy(images).float()

        # LABELS
        pad_list = zero_padding(labels)
        mask = binary_matrix(pad_list)
        mask = torch.ByteTensor(mask)
        labels = torch.LongTensor(pad_list)
        return images, labels, mask, max_label_length


    def process_image(image, height=64, channels=3):
        """Converts to self.channels, self.max_height
        # convert channels
        # resize max_height = 64
        """
        shape = image.shape
        # if shape[0] > 64 or shape[0] < 32:  # height
        try:
            image = cv2.resize(image, (int(height/shape[0] * shape[1]), height))
        except:
            return np.zeros([1, 1, channels])
        return image / 255.0


    def process_batch_images(images, height, max_width, channels=3):
        """
        Convert a list of images to a tensor (with padding)
        :param images: list of numpy array images
        :param height: desired height
        :param max_width: max width of all images
        :param channels: number of image channels
        :return: a tensor representing images
        """
        output = np.ones([len(images), height, max_width, channels])
        for i, image in enumerate(images):
            final_img = image
            shape = image.shape
            output[i, :shape[0], :shape[1], :] = final_img

        return output


    def zero_padding(l, fillvalue=PAD_token):
        """
        Pad value PAD token to l
        :param l: list of sequences need padding
        :param fillvalue: padded value
        :return:
        """
        return list(itertools.zip_longest(*l, fillvalue=fillvalue))


    def binary_matrix(l, value=PAD_token):
        m = []
        for i, seq in enumerate(l):
            m.append([])
            for token in seq:
                if token == value:
                    m[i].append(0)
                else:
                    m[i].append(1)
        return m

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
        
    import torch
    import torch.nn.functional as F
    from torch.nn import CTCLoss, CrossEntropyLoss

    def ctc_loss(outputs, targets, mask):
        USE_CUDA = torch.cuda.is_available()
        device = torch.device("cuda:0" if USE_CUDA else "cpu")
        target_lengths = torch.sum(mask, dim=0).to(device)
        # We need to change targets, PAD_token = 0 = blank
        # EOS token -> PAD_token
        targets[targets == EOS_token] = PAD_token
        outputs = outputs.log_softmax(2)
        input_lengths = outputs.size()[0] * torch.ones(outputs.size()[1], dtype=torch.int)
        loss_fn = CTCLoss(blank=PAD_token, zero_infinity=True)
        targets = targets.transpose(1, 0)
        # target_lengths have EOS token, we need minus one
        target_lengths = target_lengths - 1
        targets = targets[:, :-1]
        # print(input_lengths, target_lengths)
        torch.backends.cudnn.enabled = False
        # TODO: NAN when target_length > input_length, we can increase size or use zero infinity
        loss = loss_fn(outputs, targets, input_lengths, target_lengths)
        torch.backends.cudnn.enabled = True

        return loss, loss.item()

    import difflib

    def calculate_ac(str1, str2):
        """Calculate accuracy by char of 2 string"""

        total_letters = len(str1)
        ocr_letters = len(str2)
        if total_letters == 0 and ocr_letters == 0:
            acc_by_char = 1.0
            return acc_by_char
        diff = difflib.SequenceMatcher(None, str1, str2)
        correct_letters = 0
        for block in diff.get_matching_blocks():
            correct_letters = correct_letters + block[2]
        if ocr_letters == 0:
            acc_by_char = 0
        elif correct_letters == 0:
            acc_by_char = 0
        else:
            acc_1 = correct_letters / total_letters
            acc_2 = correct_letters / ocr_letters
            acc_by_char = 2 * (acc_1 * acc_2) / (acc_1 + acc_2)

        return float(acc_by_char)

    def accuracy_ctc(outputs, targets):
        outputs = outputs.permute(1, 0, 2)
        targets = targets.transpose(1, 0)

        total_acc_by_char = 0
        total_acc_by_field = 0

        for output, target in zip(outputs, targets):
            out_best = list(torch.argmax(output, -1))  # [2:]
            out_best = [k for k, g in itertools.groupby(out_best)]
            pred_text = get_label_from_indices(out_best)
            target_text = get_label_from_indices(target)

            # print('predict:', pred_text, 'target:', target_text)

            acc_by_char = calculate_ac(pred_text, target_text)
            total_acc_by_char += acc_by_char

            if pred_text == target_text:
                total_acc_by_field += 1

        return np.array([total_acc_by_char / targets.size()[0], total_acc_by_field / targets.size()[0]])

    from torch.utils.data import DataLoader

    def train_epoch(epoch):
        model.train()
        total_loss = 0
        total_metrics = np.zeros(2)
        for batch_idx, (images, labels, mask, max_label_length) in enumerate(train_dataloader):
            images, labels, mask = images.to(device), labels.to(device), mask.to(device)

            optimizer.zero_grad()
            output = model(images, labels, max_label_length, device)

            loss, print_loss = loss_fn(output, labels, mask)
            loss.backward()
            optimizer.step()

            total_loss += print_loss  # loss.item()
            total_metrics += metric_fn(output, labels)

            if batch_idx == len(train_dataloader):
                break

        log = {
            'loss': total_loss / len(train_dataloader),
            'metrics': (total_metrics / len(train_dataloader)).tolist()
        }

        return log

    def val_epoch():
        # when evaluating, we don't use teacher forcing
        model.eval()
        total_val_loss = 0
        # total_val_metrics = np.zeros(len(self.metrics))
        total_val_metrics = np.zeros(2)
        with torch.no_grad():
            # print("Length of validation:", len(self.valid_data_loader))
            for batch_idx, (images, labels, mask, max_label_length) in enumerate(val_dataloader):
                images, labels, mask = images.to(device), labels.to(device), mask.to(device)
                images, labels, mask = images.to(device), labels.to(device), mask.to(device)

                output = model(images, labels, max_label_length, device, training=False)
                _, print_loss = loss_fn(output, labels, mask)  # Attention:
                # loss = self.loss(output, labels, mask)
                # print_loss = loss.item()

                total_val_loss += print_loss
                total_val_metrics += metric_fn(output, labels)

        return_value = {
            'loss': total_val_loss / len(val_dataloader),
            'metrics': (total_val_metrics / len(val_dataloader)).tolist()
        }

        return return_value

    def train():
        best_val_loss = np.inf
        for epoch in range(N_EPOCHS):
            print("Epoch", epoch + 1)
            train_log = train_epoch(epoch)
            print("Training log:", train_log)
            train_loss = train_log['loss']
            train_acc_by_char = train_log['metrics'][0]
            train_acc_by_field = train_log['metrics'][1]
            experiment.log_metrics({
                "train_loss": train_loss,
                "train_acc_by_char": train_acc_by_char,
                "train_acc_by_field": train_acc_by_field
            }, epoch=epoch)

            val_log = val_epoch()
            val_loss = val_log['loss']
            val_acc_by_char = val_log['metrics'][0]
            val_acc_by_field = val_log['metrics'][1]
            experiment.log_metrics({
                "val_loss": val_loss,
                "val_acc_by_char": val_acc_by_char,
                "val_acc_by_field": val_acc_by_field
            }, epoch=epoch)
            if val_log['loss'] < best_val_loss:
                # save model
                best_val_loss = val_log['loss']

            print("Validation log:", val_log)


    def weight_reset(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            m.reset_parameters()

    # Download data from s3
    print('DOWNLOADING DATA FROM S3')
    import boto3
    import os 

    s3_resource = boto3.resource('s3')
    bucket = s3_resource.Bucket('ocrpipeline') 
    for obj in bucket.objects.filter(Prefix='data'):
        if not os.path.exists(os.path.dirname(obj.key)):
            os.makedirs(os.path.dirname(obj.key))
        bucket.download_file(obj.key, obj.key) # save to same path
            
    print('START CREATING DATASET AND MODEL')
    BATCH_SIZE = 32
    NUM_WORKERS = 4

    train_dataset = OCRDataset(data_dir='./data/train/images/', label_path='./data/train/labels.json')
    val_dataset = OCRDataset(data_dir='./data/validation/images/', label_path='./data/validation/labels.json')

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, collate_fn=collate_wrapper)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, collate_fn=collate_wrapper)

    model = CTCModel(inner_dim=128, num_chars=len(CHAR2INDEX))

    import torch
    import numpy as np

    N_EPOCHS = 50
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = ctc_loss
    metric_fn = accuracy_ctc
    device = 'cpu' if not torch.cuda.is_available() else 'cuda'

    model.apply(weight_reset)
    print('START TRAINING')
    train()

    upload_file_to_s3('best_model.pth', bucket='ocrpipeline', object_name='best_model.pth')

    model_path = "s3://ocrpipeline/best_model.pth"
    Path('/output_path.txt').write_text(model_path)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Training model')
    parser.add_argument('--data_path', type=str)

    args = parser.parse_args()

    deploy_model(data_path=args.data_path)

