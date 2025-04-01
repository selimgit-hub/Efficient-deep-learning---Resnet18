import torch
import torch.nn.utils.prune as prune
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

import models
import Factorization.fact_models

# Teacher and Student
teacher_model = models.ResNet18()
student_model = Factorization.fact_models.ResNet18()
checkpoint_student = torch.load('Lab1/Models/q_factorized.pth', map_location='cuda')
student_state_dict = checkpoint_student['net']
new_student_state_dict = {k.replace("module.", ""): v for k, v in student_state_dict.items()}
student_model.load_state_dict(new_student_state_dict)

checkpoint = torch.load('Lab1/Models/first_model.pth', map_location='cuda')
teacher_state_dict = checkpoint['net']
new_state_dict = {k.replace("module.", ""): v for k, v in teacher_state_dict.items()}
teacher_model.load_state_dict(new_state_dict)

# Move model to GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
teacher_model = teacher_model.to(device)
student_model = student_model.to(device)

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


# Define optimizer
optimizer = optim.SGD(student_model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

def distillation_loss(student_logits, teacher_logits, labels, alpha=0.6, T=3):
    criterion_ce = nn.CrossEntropyLoss()
    criterion_kl = nn.KLDivLoss(reduction = 'batchmean')

    soft_targets = F.log_softmax(student_logits/T, dim=1)
    soft_labels = F.softmax(teacher_logits/T, dim=1)
    loss_kl = criterion_kl(soft_targets, soft_labels)*(T*T)
    loss_ce = criterion_ce(student_logits, labels)

    return alpha*loss_ce + (1-alpha)*loss_kl

def train_distillation(student_model, teacher_model, train_loader, epochs=15):
    student_model.train()
    teacher_model.eval()
    for epoch in range(epochs):
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
        print(f"Epoch {epoch+1}: Loss = {running_loss/len(train_loader):.4f}")


# Train the student model with distillation
train_distillation(student_model, teacher_model, trainloader, epochs=30)

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
    print(f"Distilled Model Accuracy: {acc:.2f}%")
    return acc

# Test 
new_acc = evaluate(student_model, testloader)

# Save
state = {
    'net': student_model.state_dict(),  # Save pruned weights
    'best_acc': new_acc,  # Update with new accuracy
    'hyperparams': {
        'learning_rate': 0.01,
        'batch_size': 128,
        'momentum': 0.9,
        'weight_decay': 5e-4,
    }
}
torch.save(state, 'Lab1/Models/distilled_q_factorized.pth')

print("Distilled model saved with updated weights & accuracy!")