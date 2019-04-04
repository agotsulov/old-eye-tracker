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
        print(self.len)
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

        self.eyes = np.ndarray((self.len, 3, 64, 32))
        self.face = np.ndarray((self.len, 68, 2))
        self.x = [0 for i in range(self.len)]
        self.y = [0 for i in range(self.len)]
        print("LOADING DATA...")
        names = os.listdir('data')
        for index in range(self.len):
            curr = names[index]
            frame = misc.imread('data/' + curr)

            rects = detector(frame, 0)

            eyes_ = None

            if rects is None:
                self.len -= 1
                continue

            for (i, rect) in enumerate(rects):
                shape = predictor(frame, rect)
                shape = face_utils.shape_to_np(shape)

                (x_, y_, w_, h_) = cv2.boundingRect(np.array([shape[36:48]]))
                eyes_ = frame[y_ - 10:y_ + h_ + 10, x_ - 10:x_ + w_ + 10]

                eyes_ = cv2.resize(eyes_, (64, 32))

                self.face[index] = shape

            self.x[index] = int(curr.split('_')[2])
            self.y[index] = int(curr[:-4:].split('_')[4])

            self.eyes[index] = eyes_.reshape((3, 64, 32))

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return torch.from_numpy(self.eyes[index]).float(),\
               torch.from_numpy(self.face[index]).float(),\
               torch.from_numpy(np.array([self.x[index], self.y[index]])).float()


class ConvNet(nn.Module):
    def __init__(self, num_classes=2):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, padding=2)
        self.maxpool1 = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout2d(0.2)
        self.conv2 = nn.Conv2d(64, 256, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(2)
        self.dropout2 = nn.Dropout2d(0.2)
        self.conv3 = nn.Conv2d(256, 1024, kernel_size=3, padding=1)
        self.maxpool3 = nn.MaxPool2d(2)
        self.dropout3 = nn.Dropout2d(0.2)
        self.fc = nn.Linear(1024 * 8 * 4 + 68 * 2, num_classes)

    def forward(self, e, f):
        out = F.relu(self.conv1(e))
        out = self.maxpool1(out)
        out = self.dropout1(out)
        out = F.relu(self.conv2(out))
        out = self.maxpool2(out)
        out = self.dropout2(out)
        out = F.relu(self.conv3(out))
        out = self.maxpool3(out)
        out = self.dropout3(out)
        out = out.reshape(out.size(0), -1)
        f = f.reshape(f.size(0), -1)
        out = torch.cat((out, f), 1)
        out = self.fc(out)
        return out


# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 30
num_classes = 2
batch_size = 50
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


def train_model():
    train_dataset = Dataset()

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    model = ConvNet(num_classes).to(device)

    # Loss and optimizer
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print("TRAIN...")
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

            if (i + 1) % 10 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

    torch.save(model.state_dict(), './model.pth')

    return model

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
