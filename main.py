from pandas import read_csv
from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import numpy as np

filename = 'creditcard.csv'
df = read_csv(filename)
df1 = df.drop('Class', axis=1)
ax = df1.hist(bins=100)
for axis in ax.flatten():
    axis.set_xticklabels([])
    axis.set_yticklabels([])
plt.show()

df2 = df['Class']
a = len(df.loc[df['Class'] == 0])
b = len(df.loc[df['Class'] == 1])
c = len(df2)
x = ['Class 0 (' + str(round(a / c * 100, 2)) + '%)', 'Class 1 (' + str(round(b / c * 100, 2)) + '%)']
y = [a, b]
fig = plt.figure(figsize = (10, 5))
plt.bar(x, y, width = 0.4)
plt.ylabel("No. of transactions")
for i in range(len(x)):
    plt.text(i, y[i], y[i], ha = 'center')
plt.title("Class count")
plt.show()

class TransactionDataset(Dataset):

    def __init__(self, inputs, outputs, transform=None, target_transform=None):
        self.transactions = inputs
        self.labels = outputs

    def __len__(self):
        return len(self.transactions)

    def __getitem__(self, idx):
        transaction = self.transactions[idx]
        label = self.labels[idx]
        return transaction, label

path = "creditcard.csv"
data = read_csv(path)

X = data.drop('Class', axis=1)
y = data['Class']

train_mark = int(0.8 * len(X))

X, y = torch.tensor(X.to_numpy()).float(), torch.tensor(y.to_numpy()).float().unsqueeze(-1)

X_train, y_train = X[:train_mark], y[:train_mark]
X_test, y_test = X[train_mark:], y[train_mark:]

trainset = TransactionDataset(inputs=X_train, outputs=y_train)
testset = TransactionDataset(inputs=X_test, outputs=y_test)

batch_size = 100

train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

test_loader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

model = torch.nn.Sequential(
            torch.nn.Linear(30, 60),
            torch.nn.ReLU(),
            torch.nn.Linear(60, 30),
            torch.nn.ReLU(),
            torch.nn.Linear(30, 1),
            torch.nn.Sigmoid()
)

loss_fn = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), 1e-4)

writer = SummaryWriter('logs')

epochs = 2
train_per_epoch = int(len(trainset) / batch_size)

for e in range(epochs):
            
    loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=True)

    for idx, (transactions, labels) in loop:

        optimizer.zero_grad()
        output = model(transactions)
        loss = loss_fn(output, labels)
        loss.backward()
        optimizer.step()

        writer.add_scalar('loss', loss.item(), (e * train_per_epoch) + idx)
        accuracy = (output.round() == labels).float().mean()
        accuracy = float(accuracy)
        loop.set_description(f"Epoch [{e}/{epochs}")
        loop.set_postfix(loss=loss.item(), acc=accuracy)
        writer.add_scalar('acc', accuracy, (e * train_per_epoch) + idx)

torch.save(model, 'models/model.pth')

model.eval()
false_positives = 0
false_negatives = 0
num_correct = 0
num_samples = 0
loss_cummulative = 0

with torch.no_grad():
                
    for transactions, labels in test_loader:
                                
        output = model(transactions)
        loss = loss_fn(output, labels)
        loss_cummulative += loss.item()
        num_correct += (output.round() == labels).sum()
        num_samples += output.size(0)

        for y_pred, y_true in zip(output, labels):
            if (y_pred.round() == torch.Tensor([1.])) and (y_true == torch.Tensor([0.])):
                false_positives += 1
            elif (y_pred.round() == torch.Tensor([0.])) and (y_true == torch.Tensor([1.])):
                false_negatives += 1

print(f'Correct: {num_correct}, samples: {num_samples}, accuracy: {float(num_correct) / float(num_samples) * 100:.2f}%, false positives: {false_positives}, false negatives: {false_negatives}, loss: {loss_cummulative / len(test_loader)}')

a = round(float(num_correct) / float(num_samples) * 100, 2)
b = round(float(false_positives) / float(num_samples) * 100, 2)
c = round(float(false_negatives) / float(num_samples) * 100, 2)

x = ['Točno predviđeno (' + str(a) + '%)', 'Pogreške prve vrste (' + str(b) + '%)', 'Pogreške druge vrste (' + str(c) + '%)']
y = [int(num_correct), false_positives, false_negatives]

fig = plt.figure(figsize = (10, 5))
plt.bar(x, y, width = 0.4)
plt.ylabel("Broj transakcija")
for i in range(len(x)):
    plt.text(i, y[i], y[i], ha = 'center')
plt.title("Evaluacija modela")
plt.show()
