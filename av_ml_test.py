import torch 
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn

X = torch.tensor ([
  
  [20.0, 25.0],
  [30.0 , 12.0],
  [25.0, 12.5]
                            
                ] )
y = torch.tensor ([0.0,1.0,1.0])

#datset
dataset = TensorDataset (X, y)

#DataLoader 
loader = DataLoader (
  dataset,
  batch_size = 2,
  shuffle = True
)

#neural network
model = nn.Sequential (
  nn.Linear(2,6),
  nn.ReLU(),
  nn.Linear (6,1)

)

#Loss Function 

loss_fn = nn.MSELoss()

#optimizer
# ADAM optimization algorithms changes the weight and model hopefully gets better, it updates the parameters
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001 # learning rate
)

# Training
for epoch in range(100): #model goes through training 100 times

    for X_batch, y_batch in loader:

        prediction = model(X_batch)

        loss = loss_fn(prediction, y_batch)

        loss.backward() # Work backward from the loss and figure out how much each model parameter contributed to the error

        optimizer.step()

        optimizer.zero_grad()

print("Training complete!")