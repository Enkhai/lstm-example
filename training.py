import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader


def train(model: nn.Module, initial_state: torch.Tensor, train_loader: DataLoader, validation_loader: DataLoader,
          epochs=10, lr=0.001, clip=5, print_every=10):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for e in range(epochs):
        steps = 0

        # we initialize the hidden state to the initial one
        h = initial_state

        for x, y in train_loader:
            # set our model to training mode, we use dropout after all
            model.train()

            steps += 1

            # we get rid of the batch dimensions
            x = x.flatten(start_dim=0, end_dim=1)
            # and flatten a bit more in the case of y to reach the total batch size
            # we also need targets to be of long tensor type
            y = y.flatten(start_dim=0, end_dim=2).long()

            # we create new variables for the hidden state,
            # otherwise we would backpropagate through the entire training history
            h = tuple([each.data for each in h])

            optimizer.zero_grad()  # zero gradients
            output, h = model(x, h)  # get the new output and the hidden state
            loss = criterion(output, y)  # calculate our loss
            loss.backward()  # backpropagate
            # clip_grad_norm helps prevent the exploding gradient problem in RNNs / LSTMs
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()  # and perform a step

            # loss stats
            if steps % print_every == 0:
                # we initialize the hidden state to the initial one
                val_h = initial_state

                # we will accumulate losses to get the mean loss
                val_losses = []

                # set our model to evaluation mode, we use dropout after all
                model.eval()

                for x_, y_ in validation_loader:
                    # we get rid of the batch dimensions
                    x_ = x_.flatten(start_dim=0, end_dim=1)
                    # and flatten a bit more in the case of y to reach the total batch size
                    # we also need targets to be of long tensor type
                    y_ = y_.flatten(start_dim=0, end_dim=2).long()

                    # we create new variables for the hidden state,
                    # otherwise we would backpropagate through the entire training history
                    val_h = tuple([each.data for each in val_h])

                    # calculate the output, the hidden state and the loss
                    output, val_h = model(x_, val_h)
                    val_loss = criterion(output, y_)

                    val_losses.append(val_loss)

                print("Epoch %d/%d, step %d, loss %f, validation loss %f" % (
                    e + 1, epochs, steps, loss.item(), torch.mean(torch.tensor(val_losses)).item()))
