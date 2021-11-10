import requests
import torch
from torch import nn
from .data_processing import DataProcessing
from .model import Model


def validate(inps, tars, model_input, criterion_input):
    model_input.eval()
    val_output, val_hidden = model_input(inps)
    val_loss = criterion_input(val_output, tars.view(-1).long())
    return val_loss.item()


if __name__ == '__main__':
    # Get data
    data_path = "https://raw.githubusercontent.com/mmcky/nyu-econ-370/master/notebooks/data/book-war-and-peace.txt"
    page = requests.get(data_path)
    data = page.text.replace("\n", " ")
    chars = set(data.replace("\n", " "))

    dp = DataProcessing(data)
    train_inps, train_tars = dp.get_inps_tars(seq_len=25)
    val_inps, val_tars = dp.get_inps_tars(seq_len=25)

    # Instantiate the model
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Model(input_size=81, output_size=81, hidden_dim=24, n_layers=1)
    model.to(DEVICE)

    # Epochs and learning rate
    n_epochs = 500
    lr = 0.01

    # Define Loss, Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training Run
    train_losses = []
    val_losses = []
    for epoch in range(1, n_epochs + 1):
        model.train()
        optimizer.zero_grad()  # Clears existing gradients from previous epoch
        train_inps.to(DEVICE)
        output, hidden = model(train_inps)
        loss = criterion(output, train_tars.view(-1).long())
        loss.backward()  # Does backpropagation and calculates gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()  # Updates the weights accordingly

        if epoch % 10 == 0:
            train_losses.append(loss.item())
            val_losses.append(validate(val_inps, val_tars, model, criterion))
            print('Epoch: {}/{}.............'.format(epoch, n_epochs), end=' ')
            print("Loss: {:.4f}".format(loss.item()))
