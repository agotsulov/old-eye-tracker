import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

from imutils import face_utils
import dlib
import cv2
import numpy as np
import os
from scipy import misc

class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        self.len = len(os.listdir('data'))
        self.pretrained_face_landmark = "shape_predictor_68_face_landmarks.dat"
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(self.pretrained_face_landmark)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        names = os.listdir('data')
        curr = names[index]

        frame = misc.imread('data/' + curr)

        rects = self.detector(frame, 0)

        eyes = None
        face = None

        for (i, rect) in enumerate(rects):
            shape = self.predictor(frame, rect)
            shape = face_utils.shape_to_np(shape)

            (x_, y_, w_, h_) = cv2.boundingRect(np.array([shape[36:48]]))
            eyes = frame[y_ - 10:y_ + h_ + 10, x_ - 10:x_ + w_ + 10]

            eyes = cv2.resize(eyes, (64, 32))

            face = shape

        x = int(curr.split('_')[2])
        y = int(curr[:-4:].split('_')[4])

        # Иногда проскакивают картиники на которых dlib не может найти лицо второй раз
        print(curr)
        print(type(eyes))
        print(type(face))
        print(eyes.shape)

        return torch.from_numpy(eyes), torch.from_numpy(face), torch.from_numpy(np.array([x, y]))




# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 5
num_classes = 10
batch_size = 100
learning_rate = 0.001
'''
# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='../../data/',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='../../data/',
                                          train=False,
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)
'''

train_dataset = Dataset()

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.size(0), -1)


class ConvNet(nn.Module):
    def __init__(self, num_classes=2):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool2d(2)
        self.fc = nn.Linear(32 * 16 * 64 + 68 * 2, num_classes)

    def forward(self, e, f):
        print("FORWARD")
        print(e)
        print(f)
        print(e.size())
        print(f.size())
        out = F.relu(self.conv1(e))
        out = self.maxpool1(out)
        out = out.reshape(out.size(0), -1)
        f = f.reshape(f.size(0), -1)
        out = torch.cat(out, f)
        out = self.fc(out)
        return out


model = ConvNet(num_classes).to(device)

# Loss and optimizer
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (eyes, face, pos) in enumerate(train_loader):
        eyes = eyes.to(device)
        face = face.to(device)
        pos = pos.to(device)

        # Forward pass
        outputs = model(eyes, face)
        loss = criterion(outputs, pos)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

torch.save(model.state_dict(), 'model.ckpt')

'''
# Test the model
model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))
'''
# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')
