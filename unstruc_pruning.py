import torch
import torch.nn.utils.prune as prune
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from models import ResNet18
import torch.nn.utils.prune as prune
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


# Load ResNet model
model = ResNet18()
checkpoint = torch.load('Lab1/Models/pruned_model0.8.pth', map_location='cuda')
state_dict = checkpoint['net']

# Remove 'module.' if using DataParallel
new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
model.load_state_dict(new_state_dict)

# Move model to GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

# Preparing data
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



def unstructured_pruning(model, amount=0.3):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name="weight", amount=amount)  # Prune weights with L1 norm
            
    return model


def train(model, train_loader, epochs=15):
    model.train()
    epoch_losses = []
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
        epoch_loss = running_loss / len(train_loader)
        epoch_losses.append(epoch_loss)
        print(f"Epoch {epoch+1}: Loss = {running_loss/len(train_loader):.4f}")
    return epoch_losses


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


for prune_amount in [0.1, 0.3, 0.5, 0.7]:
    model = ResNet18()
    model.load_state_dict(new_state_dict)
    model.to(device)
    
    pruned_model = unstructured_pruning(model, amount=prune_amount)
    pruned_model = pruned_model.to(device)


    # Define optimizer
    optimizer = optim.SGD(pruned_model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    print(f"==> Finetuning with Pruning amount: {prune_amount}")
    losses = train(pruned_model, trainloader, epochs=15)
    # Remove pruning reparameterization for all Conv2d and Linear layers
    for name, module in pruned_model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if hasattr(module, 'weight_orig'):
                prune.remove(module, 'weight')
    evaluate(pruned_model, testloader)
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
            'pruning_amount': prune_amount,  # Log pruning percentage
        }
    }
    torch.save(state, f'Lab1/Models/l1_08unstruc_pruned{prune_amount}.pth')

    # Plot the losses
    plt.plot(losses, label=f'Pruning amount: {prune_amount}')


plt.title('Finetuning Losses over Epochs with Baseline Model 0.8x Pruned')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.savefig('Lab1/Plots/finetuning_losses_unstructured_baseline08.png')
plt.show()