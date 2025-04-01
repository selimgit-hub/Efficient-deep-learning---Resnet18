import torch
from collections import OrderedDict
#from models import ResNet18  # Ensure this matches your model definition
from thop import profile
from Factorization.fact_models2 import ResNet18

# Device setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load checkpoint
checkpoint = torch.load('Models/factorized_model2.pth', map_location=device)

# Extract the model state dictionary
state_dict = checkpoint['net']

# Remove 'module.' prefix if the model was trained with DataParallel
new_state_dict = OrderedDict()
for key, value in state_dict.items():
    new_key = key.replace("module.", "")  # Remove 'module.' prefix
    new_state_dict[new_key] = value

# Initialize model
model = ResNet18()

# Apply DataParallel if multiple GPUs are available
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)

# Move model to the correct device
model = model.to(device)

# Load the modified state dictionary
model.load_state_dict(new_state_dict)
model.eval()  # Set to evaluation mode

# Print loaded accuracy
print(f"Best Accuracy from Checkpoint: {checkpoint['best_acc']:.2f}%")

# Print total parameters
total_params = sum(p.numel() for p in model.parameters())
print(f'{total_params:,} total parameters.')

# Ensure dummy input is on the same device
dummy_input = torch.randn(1, 3, 32, 32).to(device)

# Compute FLOPs and parameters
flops, params = profile(model, inputs=(dummy_input,))

# Compute the model's final score
p_s, p_u = 0, 0
q_w, q_a = 32, 32
reference_w = 5.6e6
reference_f = 2.8e8

score = ((1 - (p_s + p_u)) * (q_w / 32) * params / reference_w) + ((1 - p_s) * max(q_w, q_a) / 32 * flops / reference_f)

print(f"Score of the model: {score:.6f}")
print(f"Accuracy of the model: {checkpoint['best_acc']:.2f}%")

# Print model storage size in MB
storage_size = sum(p.numel() for p in model.parameters() if p.requires_grad) * 4 / (1024 ** 2)
print(f"Model storage size: {storage_size:.2f} MB")
