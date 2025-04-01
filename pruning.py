import torch
import torch.nn.utils.prune as prune
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from models import ResNet18
import torchvision
import torchvision.transforms as transforms

def structured_pruning(model, amount=0.3):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.ln_structured(module, name="weight", amount=amount, n=2, dim=0)
    return model

# Load ResNet model
model = ResNet18()
checkpoint = torch.load('Lab1/Models/first_model.pth', map_location='cuda')
state_dict = checkpoint['net']

# Remove 'module.' if using DataParallel
new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
model.load_state_dict(new_state_dict)

# Move model to GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

# Apply pruning (e.g., 30% filter pruning)
pruned_model = structured_pruning(model, amount=0.9)
pruned_model = pruned_model.to(device)


# Training and testing loaders
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

# Define optimizer
optimizer = optim.SGD(pruned_model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

def train(model, train_loader, epochs=15):
    model.train()
    for epoch in range(epochs):
        running_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}: Loss = {running_loss/len(train_loader):.4f}")

# Train the pruned model
train(pruned_model, trainloader, epochs=15)
# Remove pruning reparameterization (for deployment)
for name, module in pruned_model.named_modules():
    if isinstance(module, nn.Conv2d):
        prune.remove(module, 'weight')

def evaluate(model, test_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    acc = 100 * correct / total
    print(f"Pruned Model Accuracy: {acc:.2f}%")
    return acc

# Test the pruned model and get new accuracy
new_acc = evaluate(pruned_model, testloader)

# Save the pruned model with updated accuracy and parameters
state = {
    'net': pruned_model.state_dict(),  # Save pruned weights
    'best_acc': new_acc,  # Update with new accuracy
    'hyperparams': {
        'learning_rate': 0.01,
        'batch_size': 128,
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'pruning_amount': 0.9,  # Log pruning percentage
    }
}
torch.save(state, 'Lab1/Models/pruned_model0.9.pth')

print("Pruned model saved with updated weights & accuracy!")
