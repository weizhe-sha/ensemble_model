import torch
import torch.nn as nn
from sklearn.ensemble import VotingClassifier as VC
from torch.nn import functional as F
from torchvision import datasets, transforms

from torchensemble import VotingClassifier
from torchensemble.utils.logging import set_logger


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Define Your Base Estimator
class NN(nn.Module):
    def __init__(self):
        super(NN,self).__init__()
        self.l1 = nn.Linear(784,500)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(500, 10)

    def forward(self,x):
        x = x.view(x.size(0), -1)
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out

class MLP(nn.Module):

    def __init__(self):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(784, 128)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, 10)

    def forward(self, data):
        data = data.view(data.size(0), -1)
        output = F.relu(self.linear1(data))
        output = F.relu(self.linear2(output))
        output = self.linear3(output)
        return output

class FinalEnsemble(nn.Module):
    def __init__(self, modelA, modelB):
        super(FinalEnsemble, self).__init__()
        self.modelA = modelA
        self.modelB = modelB

    def forward(self, x):
        x1 = self.modelA(x)
        x2 = self.modelB(x)
        outputs = [
            F.softmax(x, dim=1) for x in [x1, x2]
        ]
        # proba = op.average(outputs)
        output = sum(outputs)/2
        return output

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train = datasets.MNIST('../Dataset', train=True, download=True, transform=transform)
    test = datasets.MNIST('../Dataset', train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=128, shuffle=True)

    # Set the Logger
    logger = set_logger('classification_mnist_mlp+NN')

    # Define the ensemble
    MLP_ensemble = VotingClassifier(
        estimator=MLP,
        n_estimators=3,
        cuda=False
    )
    NN_ensemble = VotingClassifier(
        estimator=NN,
        n_estimators=3,
        cuda=False
    )

    # Set the criterion
    criterion = nn.CrossEntropyLoss()

    MLP_ensemble.set_criterion(criterion)
    NN_ensemble.set_criterion(criterion)

    # Set the optimizer
    MLP_ensemble.set_optimizer('Adam', lr=1e-3, weight_decay=5e-4)
    NN_ensemble.set_optimizer('Adam', lr=1e-3, weight_decay=5e-4)

    # Train and Evaluate
    NN_ensemble.fit(
        train_loader,
        epochs=10,
        test_loader=test_loader,
    )
    MLP_ensemble.fit(
        train_loader,
        epochs=10,
        test_loader=test_loader,
    )
MLP_ensemble = MLP()
NN_ensemble = NN()
ensemble = FinalEnsemble(MLP_ensemble, NN_ensemble)
optimizer = torch.optim.Adam(ensemble.parameters(), lr=0.001)
dataiter = iter(train_loader)
data = dataiter.next()
images, labels = data
images = images.reshape(-1, 28*28)
print(ensemble(images), labels)
for epoch in range(10):
    for i,(images, labels) in enumerate(train_loader):
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)

        outputs = ensemble(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'Epoch[{epoch+1}/{10}], Step[{i+1}/{len(train_loader)}], Loss:{loss.item():.4f}')

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = ensemble(images)

        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        for i in range(labels.size(0)):
            label = labels[i]
            pred = predicted[i]
            if label == pred:
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    acc = 100.0 * n_correct/n_samples
    print(f'Accuracy pf the ensemble network: {acc}%')

    for i in range(10):
        acc = 100.0 * n_class_correct[i]/n_class_samples[i]
        print(f'Accuracy of {[i]}:{acc}%')
#









