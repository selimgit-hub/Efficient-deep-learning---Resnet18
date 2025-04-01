import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

import models
import Factorization.fact_models_dsc
import Factorization.fact_models
import Factorization.fact_models3


# Load multiple optimized models
model_paths_dsc = [
    'Lab1/Models/quantized_dsc_sp0.5.pth',
]



model_paths_g8 = [
    'Lab1/Models/distilled_0.5pruned_q_fact3.pth',
    'Lab1/Models/distilled_0.6pruned_q_fact3.pth'
]


models_list = []
for path in model_paths_dsc:
    model = Factorization.fact_models_dsc.ResNet18()
    checkpoint = torch.load(path, map_location='cuda')
    state_dict = checkpoint['net']
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    models_list.append(model)




for path in model_paths_g8:
    model = Factorization.fact_models3.ResNet18()
    checkpoint = torch.load(path, map_location='cuda')
    state_dict = checkpoint['net']
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    models_list.append(model)   


# Move models to GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
models_list = [model.to(device) for model in models_list]

# Preparing data
print('==> Preparing data..')
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Ensemble Voting Function
def ensemble_predict(models, inputs):
    votes = torch.zeros((inputs.shape[0], 10), device=device)
    
    for model in models:
        model.eval()
        with torch.no_grad():
            outputs = model(inputs)
            votes += F.softmax(outputs, dim=1)  # Collect soft votes

    return votes.argmax(dim=1)  # Majority vote

# Evaluate Ensemble
def evaluate_ensemble(models, test_loader):
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            predicted = ensemble_predict(models, inputs)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    acc = 100 * correct / total
    print(f"Ensemble Model Accuracy: {acc:.2f}%")
    return acc

# Test the ensemble model
ensemble_acc = evaluate_ensemble(models_list, testloader)

print("Ensemble model evaluated successfully!")
