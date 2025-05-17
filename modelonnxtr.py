import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)  
        self.fc2 = nn.Linear(128, 64)       
        self.fc3 = nn.Linear(64, 10)        

    def forward(self, x):
        x = x.view(-1, 28 * 28)  
        x = F.relu(self.fc1(x))  
        x = F.relu(self.fc2(x))  
        x = self.fc3(x)          
        return x

def load_csv_input(csv_path):
    array = np.loadtxt(csv_path, delimiter=',')
    if array.size != 28 * 28:
        raise ValueError(f"Expected 784 values for a 28x28 image, got {array.size}")
    tensor = torch.tensor(array, dtype=torch.float32).reshape(1, 1, 28, 28)
    return tensor

csv_file = "csv_dataset/sample_1_28x28.csv"
test_input = load_csv_input(csv_file)

model = SimpleNet()
model.eval()

with torch.no_grad():
    output = model(test_input)

print("Model output (logits):")
print(output)

predicted_class = torch.argmax(output, dim=1)
print("Predicted class:", predicted_class.item())

torch.onnx.export(model, test_input, "model.onnx", opset_version=13)

