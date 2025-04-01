import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from models import ResNet18
import numpy as np

def l1_norm_pruning(model, prune_percentage=0.3):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # Calculate L1-norm of each filter
            filters = module.weight.data.abs().sum(dim=(1, 2, 3))
            num_filters = filters.shape[0]
            num_prune = int(prune_percentage * num_filters)
            
            # Rank filters by importance
            _, prune_indices = torch.topk(filters, num_prune, largest=False)

            # Create mask to zero out filters
            mask = torch.ones(num_filters, device=module.weight.device)
            mask[prune_indices] = 0
            
            # Apply mask to prune filters
            module.weight.data *= mask[:, None, None, None]

    return model

# Load model
model = ResNet18()
checkpoint = torch.load('Lab1/first_model.pth', map_location='cuda')
state_dict = checkpoint['net']
# Remove 'module.' if using DataParallel
new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
model.load_state_dict(new_state_dict)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Data Preparation
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

def evaluate(model):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    acc = 100 * correct / total
    print(f"Accuracy: {acc:.2f}%")
    return acc

# Prune model
model = l1_norm_pruning(model, prune_percentage=0.1)
evaluate(model)

# Retrain the model
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()

def retrain(model, epochs=15):
    model.train()
    for epoch in range(epochs):
        running_loss = 0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(trainloader):.4f}")

retrain(model, epochs=15)
evaluate(model)

# Save the model
state = {
    'net': model.state_dict(),
    'best_acc': evaluate(model),
    'hyperparams': {
        'learning_rate': 0.01,
        'batch_size': 128,
        'momentum': 0.9,
        'weight_decay': 5e-4,
    }
}
torch.save({'net': model.state_dict()}, 'Lab1/l1_norm_pruned_model0.1.pth')
print("Model Pruned and Saved")
