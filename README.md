# Stock-Price-Prediction


## AIM

To develop a Recurrent Neural Network model for stock price prediction.

## Problem Statement and Dataset
The objective of this project is to develop a Stock Price Prediction system using a Recurrent Neural Network (RNN). Stock price prediction is an important task in financial analysis because it helps investors and analysts make better decisions based on historical trends. Since stock prices are time-series data, RNN models are suitable because they can capture sequential patterns and dependencies in past observations. In this project, the RNN model is trained to learn patterns from historical stock prices and predict future prices based on previous time steps.

The dataset used for this project consists of historical stock market data, which is divided into two CSV files: trainset.csv and testset.csv. The training dataset is used to train the model, while the testing dataset is used to evaluate the model’s performance. Each dataset typically contains columns such as Date, Open, High, Low, Close, and Volume, but in this implementation only the Close (closing price) column is used for prediction. The data is first normalized using MinMaxScaler, and then converted into sequences of 60 time steps so that the RNN can learn temporal relationships in the data.

## Design Steps

## Step 1:
Import necessary libraries.

## Step 2:
Load and preprocess the data.

## Step 3:
Create input-output sequences.

## Step 4:
Convert data to PyTorch tensors.

## Step 5:
Define the RNN model.

## Step 6:
Train the model using the training data

## Step 7:
Evaluate the model and plot predictions.



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

The RNN model successfully learned patterns from historical stock data and predicted future stock prices with reasonable accuracy, closely following the trend of the actual prices.
