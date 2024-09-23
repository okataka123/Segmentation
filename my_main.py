from re import T
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from models.UNet import *


# データの変換（データオーグメンテーションや正規化など）
transform = transforms.Compose([
#   transforms.Resize((224, 224)),
    transforms.ToTensor(),  # 画像をTensorに変換
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 正規化
])

# Pascal VOC データセットをロード
train_dataset = datasets.VOCSegmentation(
    root='~/datasets',  # データを保存するディレクトリ
    year='2012',  # '2012' または '2007'
    image_set='train',  # 'train', 'val', 'trainval' など
    download=False,  # データを自動ダウンロード
    transform=transform  # 画像に適用する変換
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

# train
model = UNet()
criterion = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)


# start training
best_loss = 10**5
epochs = 20

for i in range(5):
    image, target = train_dataset[i]
    print(f"Image {i}: {image.shape}, Target: {target.size}")
    image_viz = image.permute(1, 2, 0)
    plt.imshow(image_viz)
    plt.show()
    plt.imshow(target)
    plt.show()



# for epoch in tqdm(range(epochs)):
#     print(f'==========Epoch {epoch+1} Start Training==========')
#     model.train()
#     train_loss = 0
    
#     for images, targets in train_loader:
#         print(images)
#         import pdb; pdb.set_trace()