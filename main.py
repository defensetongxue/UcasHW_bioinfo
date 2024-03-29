import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import os
import matplotlib
matplotlib.use('Agg')  # Use the Anti-Grain Geometry non-GUI backend
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

class ProteinDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.files = os.listdir(root_dir)
        self.AA_DICT = {
            'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4,
            'Q': 5, 'E': 6, 'G': 7, 'H': 8, 'I': 9,
            'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14,
            'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19
        }
        self.SSP_DICT = {'C': 0, 'H': 1, 'E': 2}

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file_path = os.path.join(self.root_dir, self.files[idx])
        with open(file_path, 'rb') as f:
            sample = pickle.load(f)
        seq, ssp = sample['seq'], sample['ssp']
        seq_encoded = self.encode_sequence(seq).unsqueeze(2)
        ssp_encoded = self.encode_ssp(ssp)
        return seq_encoded, ssp_encoded
    
    def encode_sequence(self, seq):
        encoded = torch.zeros((len(seq), 20), dtype=torch.float32)
        for i, aa in enumerate(seq):
            if aa in self.AA_DICT:
                encoded[i, self.AA_DICT[aa]] = 1
        return encoded

    def encode_ssp(self, ssp):
        encoded = torch.tensor([self.SSP_DICT[s] for s in ssp if s in self.SSP_DICT], dtype=torch.long)
        return encoded

class ProteinCNN(nn.Module):
    def __init__(self):
        super(ProteinCNN, self).__init__()
        self.fc1 = nn.Linear(20, 1024,bias=False)
        self.conv1 = nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=1024, out_channels=128, kernel_size=3, padding=1)
        self.fc = nn.Linear(128, 3)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x)).unsqueeze(2)
        x = F.relu(self.conv1(x)+x)
        x = F.relu(self.conv2(x)).squeeze(2)
        x = self.fc(x)
        return x

class ProteinMLP(nn.Module):
    def __init__(self, input_size, hidden_size=1024):
        super(ProteinMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 3)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_and_evaluate_model(model, dataset, epochs=10, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    kf = KFold(n_splits=3)
    fold_results = []

    for fold, (train_ids, test_ids) in enumerate(kf.split(dataset)):
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

        train_loader = DataLoader(dataset, batch_size=1, sampler=train_subsampler)
        test_loader = DataLoader(dataset, batch_size=1, sampler=test_subsampler)

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for seq, ssp in train_loader:
                seq = seq.squeeze(0)  # Adjust for CNN input
                ssp = ssp.squeeze()
                optimizer.zero_grad()
                outputs = model(seq)
                loss = criterion(outputs, ssp)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            
            print(f"Fold {fold+1}, Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")
            fold_results.append(running_loss/len(train_loader))
           
        # Model evaluation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for seq, ssp in test_loader:
                seq = seq.squeeze(0)  # Adjust for CNN input
                ssp = ssp.squeeze()
                outputs = model(seq)
                _, predicted = torch.max(outputs.data, 1)
                total += ssp.size(0)
                correct += (predicted == ssp).sum().item()
        accuracy = correct / total
        print(f"Fold {fold+1} Accuracy: {accuracy * 100}%")
        break
    return fold_results

def plot_loss_curve(losses,save_name):
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label='Training Loss')
    plt.title('Loss Curve across folds')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'./{save_name}.png')

if __name__ == '__main__':
    dataset = ProteinDataset('./assignment1_data')
    model_cnn = ProteinCNN()
    model_mlp = ProteinMLP(input_size=20)  # Assuming all sequences are of same length

    print("Training CNN...")
    losses_cnn = train_and_evaluate_model(model_cnn, dataset, epochs=50, lr=1e-3)
    print("Training MLP...")
    losses_mlp = train_and_evaluate_model(model_mlp, dataset, epochs=50, lr=1e-3)

    # Assuming you want to plot the losses for the last fold
    plot_loss_curve(losses_cnn,'cnn')
    plot_loss_curve(losses_mlp,'mlp')
