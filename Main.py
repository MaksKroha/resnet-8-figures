import torch

from DataLoader import Loader
from Model import ResNet

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # hyper parameters
    batch_size = 512
    learning_rate = 0.0001
    epochs = 1
    dropout_prob = 0.3
    train = False

    # testing parameters
    # programmer choose which block to lock during testing
    first_blk_act = True
    second_blk_act = True
    third_blk_act = True
    train_device = "cuda"
    test_device = "cpu"


    data_loader = Loader(batch_size)
    train_loader = data_loader.getTrainLoader()
    test_loader = data_loader.getTestLoader()

    model = ResNet(learning_rate, dropout_prob)
    model = model.to(train_device)

    # training mode
    # trying to load previous weights if exists
    try:
        state_dict = torch.load("parameters/model_state_dict.pt", weights_only=True)
        model.load_state_dict(state_dict)
    except(FileNotFoundError, EOFError): pass

    if train:
        print("Training starts")
        model.train()
        for epoch in range(epochs):
            for images, labels in train_loader:
                images, labels = images.to(train_device), labels.to(train_device)
                logits = model(images)
                model.backward(logits, labels)
            print(f"{epoch + 1}/{epochs} epoch has been passed")
        torch.save(model.state_dict(), "parameters/model_state_dict.pt")


    # testing mode
    losses = [100]
    x = [0]
    model.eval()
    model.to(test_device)
    with torch.no_grad():
        print("Testing starts")
        for idx, (image, label) in enumerate(test_loader):
            logits = model(image, first_blk_act, second_blk_act, third_blk_act)
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            x.append(idx + 1)
            losses.append(100 - probabilities[0][label.item()] * 100)
    plt.scatter(x, losses, label="Похибка в %")
    plt.xlabel("index")
    plt.ylabel("loss")
    plt.show()



