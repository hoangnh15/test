import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GatedGraphConv, global_mean_pool
from torch_geometric.data import DataLoader
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from collections import Counter

# GGNN Model
class GGNNWithLocalGlobal(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers):
        super(GGNNWithLocalGlobal, self).__init__()
        self.ggnn = GatedGraphConv(hidden_dim, num_layers)
        self.local_fc = nn.Linear(hidden_dim, hidden_dim)  # Fully connected for local features
        self.global_fc = nn.Linear(hidden_dim, num_classes)  # Fully connected for global features

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Local embedding through GGNN
        x = self.ggnn(x, edge_index)
        local_feature = F.relu(self.local_fc(x))  # Local feature representation

        # Global embedding through pooling
        global_feature = global_mean_pool(local_feature, batch)  # Mean pooling for global feature
        out = self.global_fc(global_feature)  # Classification
        return F.log_softmax(out, dim=-1)



# Load dataset
dataset = torch.load('graph_dataset_01.pt')

# Count samples by label
def count_samples_by_label(dataset):
    label_counts = Counter()
    for data in dataset:
        label = int(data.y.item())  # assuming data.y is a tensor containing the label
        label_counts[label] += 1
    return label_counts

label_counts = count_samples_by_label(dataset)
print("Số lượng mẫu trên từng nhãn:", label_counts)

# Get class weights based on label frequency
total_samples = sum(label_counts.values())
class_weights = {label: total_samples / count for label, count in label_counts.items()}
print("Trọng số các lớp:", class_weights)

# Convert class weights to tensor for loss function
weights_tensor = torch.tensor([class_weights[label] for label in sorted(class_weights.keys())], dtype=torch.float)
print("Tensor trọng số:", weights_tensor)

# Split dataset into train and test sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Initialize model, optimizer, and loss function
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GGNNWithLocalGlobal(input_dim=16, hidden_dim=128, num_classes=4, num_layers=4).to(device)  # Increased hidden_dim and num_layers
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Reduced learning rate
criterion = torch.nn.CrossEntropyLoss(weight=weights_tensor.to(device))

# Training function
def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()  # Accumulate loss
    avg_loss = total_loss / len(train_loader)  # Calculate average loss for this epoch
    return avg_loss

# Testing function
def test(loader):
    model.eval()
    correct = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            pred = out.argmax(dim=1)
            correct += (pred == data.y).sum().item()
            y_true.extend(data.y.tolist())
            y_pred.extend(pred.tolist())
    accuracy = correct / len(loader.dataset)
    return accuracy, y_true, y_pred

# Training loop
for epoch in range(1, 101):
    train_loss = train()
    test_acc, _, _ = test(test_loader)
    print(f"Epoch {epoch}, Loss: {train_loss:.4f}, Test Accuracy: {test_acc:.4f}")

# Test and evaluate final model
test_acc, y_true, y_pred = test(test_loader)
print(classification_report(y_true, y_pred, target_names=['Normal', 'Overflow', 'Selfdestruct', 'Timestamp']))

# Confusion Matrix
class_names = ['Normal', 'Overflow', 'Selfdestruct', 'Timestamp']

def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    ax.set_title('Confusion Matrix')
    plt.tight_layout()

    plt.show()

# Plot confusion matrix
plot_confusion_matrix(y_true, y_pred, class_names)
