import os
from Utils.basic_cnn import CNNMusicNet
from Utils.cnn_trainer import Trainer
import torchvision
import torch

data_path = 'C:\\Users\\Teddy\\Documents\\Academics\\Machine Learning\\Projects\\CS_4641_Project\\src\\musicNet\\data'
orig_set = torchvision.datasets.ImageFolder(
    root=data_path + '\\midi_rolls',
    transform=torchvision.transforms.ToTensor()
)
n = len(orig_set)  # total number of examples
n_test = int(0.1 * n)  # take ~10% for test
test_set = torch.utils.data.Subset(orig_set, range(n_test))  # take first 10%
train_set = torch.utils.data.Subset(orig_set, range(n_test, n))  # take the rest   

trainloader = torch.utils.data.DataLoader(
    train_set,
    batch_size=64,
    num_workers=0,
    shuffle=True
)
testloader = torch.utils.data.DataLoader(
    test_set,
    batch_size=64,
    num_workers=0,
    shuffle=True
)

cnn = CNNMusicNet()

trainer = Trainer(cnn, trainloader, testloader, num_epochs=500, batch_size=32, init_lr=1e-3, device="cpu")
trainer.train()

classes = set(os.listdir(data_path + '/musicnet_midis'))
print(classes)

# Choose best device to speed up training
device = ("cuda" if torch.cuda.is_available() 
          else "cpu")
print(f"Using {device} device")

