import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
import os

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
# from matplotlib.font_manager import FontProperties

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

def load_custom_image_dataset(data_path):
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor()
    ])
    dataset = datasets.ImageFolder(root=data_path, transform=transform)
    return dataset

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

def distribute_data_to_clients(dataset, num_clients):
    num_samples_per_client = len(dataset) // num_clients
    split_dataset = [num_samples_per_client] * (num_clients - 1)
    split_dataset.append(len(dataset) - num_samples_per_client * (num_clients - 1))
    return random_split(dataset, split_dataset)

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
    
def main(EPOCH, BATCH_SIZE, CLIENT_NUM, DATA_PATH, epsilon):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # Create dataset
    dataset = load_custom_image_dataset(DATA_PATH)
    num_classes = len(dataset.classes)
    print(f'Number of classes in {DATA_PATH}: {num_classes}')
    
    # Create && split dataloader
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # 分配数据给客户端
    client_datasets = distribute_data_to_clients(train_dataset, CLIENT_NUM)

    # 初始化全局模型
    global_model = ConvNet(num_classes).to(device)
    global_optimizer = optim.SGD(global_model.parameters(), lr=0.001)
    
    # 训练设置
    num_epochs = EPOCH
    batch_size = BATCH_SIZE

    history_test = []
    history_train = []

    for epoch in range(num_epochs):
        global_model.train()
        global_optimizer.zero_grad()

        # Initialize gradients
        for param in global_model.parameters():
            param.grad = torch.zeros_like(param.data)

        for client_data in client_datasets:
            local_model = ConvNet(num_classes).to(device)
            local_model.load_state_dict(global_model.state_dict())
            local_optimizer = optim.SGD(local_model.parameters(), lr=0.001)
            train_loader = DataLoader(client_data, batch_size=batch_size, shuffle=True)

            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                local_optimizer.zero_grad()
                output = local_model(data)
                loss = nn.CrossEntropyLoss()(output, target)
                loss.backward()
                local_optimizer.step()

            # Aggregate gradients
            for global_param, local_param in zip(global_model.parameters(), local_model.parameters()):
                if global_param.grad is None:
                    global_param.grad = local_param.grad.clone()
                else:
                    global_param.grad += local_param.grad

        # Average the gradients
        for param in global_model.parameters():
            param.grad /= len(client_datasets)
            param.grad -= epsilon
            
        # Update global model
        global_optimizer.step()

        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        evaluate_model(global_model, device, test_loader, epoch, history_test)

    # 保存全局模型
    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')
    torch.save(global_model.state_dict(), f"saved_models/global_model_{epoch}_{epsilon}.pth")
    print("Training complete and model saved!")
    
    # 視覺化訓練紀錄
    visHistory(history=history_test, epsilon=epsilon)

if __name__ == '__main__':
    # modify -> EPOCH, BATCH, CLIENT, DATA_PATH, epsilon
    # TO-BE done: epsilon
    EPOCH = [10, 50, 100, 200]
    BATCH = 32
    CLIENT = 3
    DATA_PATH = "image_data"
    epsilon = [1000, 0, 0.00001, 1e-6, 1e-9]
    
    for epoch in EPOCH:
        main(EPOCH=epoch, BATCH_SIZE=BATCH, CLIENT_NUM=CLIENT, DATA_PATH=DATA_PATH, epsilon=epsilon[3])