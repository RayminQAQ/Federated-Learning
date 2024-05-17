# pipline 

# package dependency
import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
import numpy as np
# from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

class ConvNet(nn.Module):
    def __init__(self, num_classes):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Create dataset via ImageFolder
def load_custom_image_dataset(data_path):
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    dataset = datasets.ImageFolder(root=data_path, transform=transform)
    return dataset

def distribute_data_to_clients(dataset, num_clients):
    num_samples_per_client = len(dataset) // num_clients
    split_dataset = [num_samples_per_client] * (num_clients - 1)
    split_dataset.append(len(dataset) - num_samples_per_client * (num_clients - 1))
    return random_split(dataset, split_dataset)

# Some tools to evaluate model's Accuracy
def evaluate_model(model, device, dataloader, epoch, history):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    accuracy = 100 * correct / total
    print(f'Epoch: {epoch} and Accuracy of the model on the test images: {accuracy} %')
    
    # Store into history
    history.append(accuracy)

# Some tools to visualize model's Accuracy
def visHistory(history, epsilon):
    epochs = list(range(1, len(history) + 1))

    plt.title(f"Federating Learning in {len(history)} epoch (epsilon={epsilon})")
    plt.plot(epochs, history, 'b-o')  # 假设history是准确度数据
    
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    
    if not os.path.exists('history'):
        os.makedirs('history')
    plt.savefig(f'history/learning_history_{len(history)}.png')    
    plt.close()
    
    print(f"History saved in 'history/learning_history_{len(history)}.png")

def client_update(client_model, optimizer, train_loader, epoch, device):
    """
    This function updates/trains client model on client data
    """
    client_model.train()
    for e in range(epoch):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = client_model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()
            optimizer.step()
    return loss.item()
 
def server_aggregate(global_model, client_models, epsilon):
    """
    This function has aggregation method 'mean'
    """
    ### This will take simple mean of the weights of models ###
    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        global_dict[k] = torch.stack([client_models[i].state_dict()[k].float() for i in range(len(client_models))], 0).mean(0)
        global_dict[k] += epsilon
    global_model.load_state_dict(global_dict)
    for model in client_models:
        model.load_state_dict(global_model.state_dict())
 
def main(ROUNDS, CLIENT_EPOCH, BATCH_SIZE, CLIENT_NUM, CLIENT_SELECT, DATA_PATH, epsilon):
    # Cuda support
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # Create dataset
    dataset = load_custom_image_dataset(DATA_PATH)
    num_classes = len(dataset.classes)
    print(f'Number of classes in {DATA_PATH}: {num_classes}')
    
    # Split dataset into train and test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Split train dataset into multiple clients
    client_datasets = distribute_data_to_clients(train_dataset, CLIENT_NUM)

    # Dataloader
    client_dataloaders = []
    for i in range(CLIENT_NUM):
        client_dataloader = DataLoader(client_datasets[i], batch_size=BATCH_SIZE, shuffle=True)
        client_dataloaders.append(client_dataloader)
        
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize server and client models
    server_model = ConvNet(num_classes).to(device)    
    client_models = [ConvNet(num_classes).to(device) for _ in range(CLIENT_NUM)]
    for model in client_models:
        model.load_state_dict(server_model.state_dict())
    
    # server_optimizer = optim.Adam(server_model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    client_optimizer = [optim.Adam(server_model.parameters(), lr=0.1, betas=(0.9, 0.999), eps=1e-08, weight_decay=0) for model in client_models]
    
    # some parameter for tmp
    history_test = []
    history_train = []

    for round in range(ROUNDS):
        # select random clients
        client_idx = np.random.permutation(CLIENT_NUM)[:CLIENT_SELECT]

        # client update
        for i in range(CLIENT_SELECT):
            # client_syn(client_models[i], server_model)
            loss = client_update(client_models[i], client_optimizer[i], client_dataloaders[client_idx[i]], CLIENT_EPOCH, device)
            print(f"Client({i}) loss is: {loss}")
        
        # server aggregate
        server_aggregate(server_model, client_models, epsilon)
        evaluate_model(server_model, device, test_loader, round, history_test)

    # save server model
    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')
    torch.save(server_model.state_dict(), f"saved_models/global_model_{ROUNDS}_{epsilon}.pth")
    print("Training complete and model saved!")
    
    # vis
    visHistory(history=history_test, epsilon=epsilon)

if __name__ == '__main__':
    # modify -> EPOCH, BATCH, CLIENT, DATA_PATH, epsilon
    # TO-BE done: epsilon
    ROUNDS = [10, 50, 100, 200]
    EPOCH = 10
    BATCH = 32
    CLIENT_TOTAL = 3
    CLIENT_SELECT = 2
    DATA_PATH = "image_fromTA"
    epsilon = [1000, 0, 0.00001, 1e-6, 1e-9]
    
    for round in ROUNDS:
        main(ROUNDS=round, CLIENT_EPOCH=EPOCH, BATCH_SIZE=BATCH, CLIENT_NUM=CLIENT_TOTAL, CLIENT_SELECT=CLIENT_SELECT,  DATA_PATH=DATA_PATH, epsilon=epsilon[1])