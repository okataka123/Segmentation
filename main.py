import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import argparse 
from tqdm import tqdm
import matplotlib.pyplot as plt
from models.UNet import *


# データの変換（データオーグメンテーションや正規化など）
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),  # 画像をTensorに変換
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 正規化
])

# マスク画像に対するtransform
mask_transform = transforms.Compose([
    transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST),
    transforms.ToTensor(),  # 画像をTensorに変換
])


# Pascal VOC データセットをロード
train_dataset = datasets.VOCSegmentation(
    root='~/datasets',  # データを保存するディレクトリ
    year='2012',  # '2012' または '2007'
    image_set='train',  # 'train', 'val', 'trainval' など
    download=False,  # データを自動ダウンロード
    transform=transform,  # 画像に適用する変換
    target_transform=mask_transform,
)

test_dataset = datasets.VOCSegmentation(
    root='~/datasets',
    year='2012',
    image_set='val',
    download=False,
    transform=transform
)

# DataLoaderを作成して、バッチごとにデータをロード
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=8,  # バッチサイズ
    shuffle=True,  # データをシャッフル
#    num_workers=4  # データ読み込みの並列処理
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=8,
    shuffle=False,
#    num_workers=4
)


def vis_dataset():
    # datasetの中身を確認
    for i in range(5):
        image, target = train_dataset[i]
        #print(f"Image {i}: {image.shape}, Target: {target.shape}")
        image_viz = image.permute(1, 2, 0)
        target = target.permute(1, 2, 0)
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(image_viz)
        ax[1].imshow(target)
        plt.show()
#vis_dataset()

# DataLoaderの動きを確認
def vis_dataloader():
    for images, targets in train_loader:
        print('len(images) =', len(images))
        print('len(targets) =', len(targets))
        print(f"Image: {images[0].shape}, Target: {targets[0].size}")
        assert False
#vis_dataloader()

def visualization_loss(train_loss_value):
    '''
    epochごとのtrain lossとtest lossの推移グラフを可視化
    '''
    plt.figure(figsize=(6,6))
    plt.plot(range(len(train_loss_value)), train_loss_value, label='train loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.show()

def train(savemodel=False):
    model = UNet(n_channels=3, n_classes=21) # Pascal VOCの場合
    #criterion = torch.nn.BCELoss()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    epochs = 20

    train_loss_value = []
    for epoch in tqdm(range(epochs)):
        print(f'==========Epoch {epoch+1} Start Training==========')
        model.train()
        train_loss = 0
        for images, targets in tqdm(train_loader, total=len(train_loader)):
            optimizer.zero_grad()
            outputs = model(images)
            

            # print('outputs[0].shape =', outputs[0].shape)
            # print('targets[0].shape =', targets[0].shape)

            # print('targets[0] =', targets[0])

            # print('outputs.shape =', outputs.shape)
            # print('targets.shape =', targets.shape)

            targets_2 = targets.squeeze(1)
            targets_2 = targets_2.long()
            # print('targets_2.shape =', targets_2.shape)
            # print('set(targets_2) =', set(targets_2))

            #loss = criterion(outputs, targets)
            loss = criterion(outputs, targets_2)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss_value.append(train_loss/len(train_loader))
    visualization_loss(train_loss_value)
    if savemodel:
        pass


# 推論
def inference():
    pass


def main():
    parser = argparse.ArgumentParser(description='Train or run inference on the model.')
    parser.add_argument('--train', action='store_true', help="Run training mode.")
    parser.add_argument('--inference', action='store_true', help="Run inference mode.")
    
    args = parser.parse_args()
    if args.train and args.inference:
        print("Error: Please specify either --train or --inference, not both.")
    elif args.train:
        train(savemodel=True)
    elif args.inference:
        inference()
    else:
        print("Error: Please specify either --train or --inference.")

if __name__ == '__main__':
    main()