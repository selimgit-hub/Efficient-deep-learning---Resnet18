import torch
import torch.nn.utils.prune as prune
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import time
from fvcore.nn import FlopCountAnalysis
#from models import ResNet18
#from models import ResNet18
#from models import ResNet18
from Factorization.fact_models3 import ResNet18
#from Factorization.fact_models import ResNet18

# Load ResNet model
model = ResNet18()
checkpoint = torch.load('Lab1/Models/distilled_0.6pruned_q_fact3.pth', map_location='cuda')
state_dict = checkpoint['net']

# Remove 'module.' if using DataParallel
new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
model.load_state_dict(new_state_dict,strict=False)

# Move model to GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

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
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

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
    print(f"Model Accuracy: {acc:.2f}%")
    return acc


def count_nonzero_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    nonzero_params = sum(torch.count_nonzero(p).item() for p in model.parameters())
    print(f"Total params: {total_params}, Non-zero params after pruning: {nonzero_params}")
    return nonzero_params


def measure_latency(model, input_size=(1, 3, 32, 32), runs=50):
    model.eval()
    dummy_input = torch.randn(input_size).to(device)

    # Warm-up
    for _ in range(10):
        _ = model(dummy_input)

    # Measure
    start_time = time.time()
    for _ in range(runs):
        _ = model(dummy_input)
    end_time = time.time()

    avg_latency = (end_time - start_time) / runs
    print(f"Avg Inference Latency: {avg_latency:.6f} sec")
    return avg_latency

baseline_latency = measure_latency(model)


# Flops
flops = FlopCountAnalysis(model, torch.randn(1, 3, 32, 32).to(device))
flops = flops.total()


evaluate(model, testloader)
params = count_nonzero_params(model)
print(f"FLOPs: {flops}")

p_s, p_u = 0.6, 0
q_w, q_a = 16, 16
reference_w = 5.6e6
reference_f = 2.8e8

score = ((1 - (p_s + p_u)) * (q_w / 32) * params / reference_w) + ((1 - p_s) * max(q_w, q_a) / 32 * flops / reference_f)
print(f"Score of the model: {score:.6f}")