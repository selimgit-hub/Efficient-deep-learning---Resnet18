import torch
import torch.nn.utils.prune as prune
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms

import models
#import Factorization.fact_models_dsc
import Factorization.fact_models3

# Function for structured pruning
def structured_pruning(model, amount=0.3, prune_type="conv"):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.ln_structured(module, name="weight", amount=amount, n=2, dim=0)
    return model

# Load Teacher and Student Models
teacher_model = models.ResNet18()
student_model = Factorization.fact_models3.ResNet18()

# Load pretrained teacher model
checkpoint_teacher = torch.load('Lab1/Models/first_model.pth', map_location='cuda')
teacher_state_dict = checkpoint_teacher['net']
teacher_model.load_state_dict({k.replace("module.", ""): v for k, v in teacher_state_dict.items()})

# Load student model
checkpoint_student = torch.load('Lab1/Models/q_factorised3.pth', map_location='cuda')
student_state_dict = checkpoint_student['net']
student_model.load_state_dict({k.replace("module.", ""): v for k, v in student_state_dict.items()})

# Move models to GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
teacher_model = teacher_model.to(device)
student_model = student_model.to(device)

# **Apply Structured Pruning to Student Model**
pruned_student_model = structured_pruning(student_model, amount=0.6, prune_type="conv")
pruned_student_model = pruned_student_model.to(device)

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

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# Define optimizer
optimizer = optim.SGD(pruned_student_model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

# Distillation loss function
def distillation_loss(student_logits, teacher_logits, labels, alpha=0.6, T=3):
    criterion_ce = nn.CrossEntropyLoss()
    criterion_kl = nn.KLDivLoss(reduction='batchmean')

    soft_targets = F.log_softmax(student_logits/T, dim=1)
    soft_labels = F.softmax(teacher_logits/T, dim=1)
    loss_kl = criterion_kl(soft_targets, soft_labels) * (T*T)
    loss_ce = criterion_ce(student_logits, labels)

    return alpha * loss_ce + (1 - alpha) * loss_kl

# Training with distillation
def train_distillation(student_model, teacher_model, train_loader, epochs=15):
    student_model.train()
    teacher_model.eval()

    
    for epoch in range(epochs):
        train_loss = 0
        
        running_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            with torch.no_grad():
                teacher_outputs = teacher_model(inputs)

            student_outputs = student_model(inputs)
            loss = distillation_loss(student_outputs, teacher_outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss += loss.item()
        
        print(f"Epoch {epoch+1}: Loss = {running_loss/len(train_loader):.4f}")

# Train pruned student model with distillation
train_distillation(pruned_student_model, teacher_model, trainloader, epochs=30)

# Remove pruning for deployment
for name, module in pruned_student_model.named_modules():
    if isinstance(module, nn.Conv2d):
        prune.remove(module, 'weight')

# Evaluation function
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
    print(f"Distilled & Pruned Model Accuracy: {acc:.2f}%")
    return acc

# Evaluate the pruned and distilled student model
new_acc = evaluate(pruned_student_model, testloader)

# Save the final distilled + pruned student model
state = {
    'net': pruned_student_model.state_dict(),
    'best_acc': new_acc,
    'hyperparams': {
        'learning_rate': 0.01,
        'batch_size': 128,
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'pruning_amount': 0.6,  # Log pruning percentage
    }
}
torch.save(state, 'Lab1/Models/distilled_0.6pruned_q_fact3.pth')


