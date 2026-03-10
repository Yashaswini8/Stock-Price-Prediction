# Stock-Price-Prediction


## AIM

To develop a Recurrent Neural Network model for stock price prediction.

## Problem Statement and Dataset


## Design Steps

### Step 1:
Write your own steps

### Step 2:

### Step 3:



## Program
#### Name:
#### Register Number:
Include your code here
```Python 
# Define RNN Model
class RNNModel(nn.Module):

    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(RNNModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.rnn(x, h0)

        out = self.fc(out[:, -1, :])

        return out





model = RNNModel()

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# Train the Model

num_epochs = 20

train_losses = []

for epoch in range(num_epochs):

    model.train()
    epoch_loss = 0

    for X_batch, y_batch in train_loader:

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        outputs = model(X_batch)

        loss = criterion(outputs, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    epoch_loss = epoch_loss / len(train_loader)
    train_losses.append(epoch_loss)

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.6f}')
print('Name: Yashaswini S                ')
print('Register Number:212224220123     ')
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.show()






```

## Output

### True Stock Price, Predicted Stock Price vs time


<img width="829" height="702" alt="image" src="https://github.com/user-attachments/assets/1447570a-dcd3-45f0-835f-2343c32338bf" />


### Predictions 

<img width="877" height="518" alt="image" src="https://github.com/user-attachments/assets/d6cc5588-b2ce-4f35-8c44-2a74a1bf6382" />


## Result


