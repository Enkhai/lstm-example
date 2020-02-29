import numpy as np
import torch
from models import CharRNN
import torch.nn.functional as F


def sample(model: CharRNN, char2int: dict, prime='The', num_chars=1000, top_k=5):
    """
    Given a network and a char2int map, predict the next 1000 characters
    """

    device = next(model.parameters()).device.type

    int2char = {ii: ch for ch, ii in char2int.items()}

    # set our model to evaluation mode, we use dropout after all
    model.eval()

    # First off, run through the prime characters
    chars = [char2int[ch] for ch in prime]
    h = model.init_hidden(1, device)
    for ch in chars:
        char, h = predict(model, ch, h, top_k, device)

    chars.append(char)

    # Now pass in the previous character and get a new one
    for ii in range(num_chars):
        char, h = predict(model, chars[-1], h, top_k, device)
        chars.append(char)

    return ''.join(int2char[c] for c in chars)


def predict(model: CharRNN, char, h, top_k=5, device="cpu"):
    """
    Given an integer encoded character, a hidden state and a network, predict the next integer encoded character.
    Returns the predicted character and the new hidden state of the network.
    """

    # tensor inputs
    inputs = one_hot_encode_tensor(torch.tensor([[char]]), model.num_chars).to(device)

    # detach hidden state from history
    h = tuple([each.data for each in h])
    # get the output of the model
    out, h = model(inputs, h)

    # get the character probabilities
    p = F.softmax(out, dim=1).data.to(device)

    # get top characters
    p, top_ch = p.topk(top_k)
    p, top_ch = p.cpu().numpy().squeeze(), top_ch.cpu().numpy().squeeze()

    # select the likely next character with some element of randomness
    char = np.random.choice(top_ch, p=p / p.sum())

    # return the encoded value of the predicted char and the hidden state
    return char, h


def count_parameters(model: CharRNN):
    """
    counts the total number of parameters in a model
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def validation_split(data, val_frac=0.1):
    """
    splits string data into training and validation sets
    """
    val_idx = int(len(data) * (1 - val_frac))
    return data[:val_idx], data[val_idx:]


def one_hot_encode(arr, n_labels):
    """
    for n labels and a 2-D arr of integers,
    return a one-hot 2-D numpy array representation
    """
    # create an initial one-hot array filled with zeros with the size of the array and the number of labels
    one_hot = np.zeros((arr.size, n_labels), dtype=np.float32)

    # fill in the appropriate elements with ones
    # arr as we already mentioned is 2-D array so we flatten it
    one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1

    # finally, we reshape to get back to the original array
    one_hot = one_hot.reshape((*arr.shape, n_labels))

    return one_hot


def one_hot_encode_tensor(tensor: torch.Tensor, n_labels):
    """
    for n labels and a 2-D tensor,
    return a one-hot 2-D tensor representation
    """
    one_hot = torch.zeros((tensor.nelement(), n_labels), dtype=torch.float)
    one_hot[torch.arange(one_hot.shape[0]), tensor.flatten().long()] = 1
    one_hot = one_hot.reshape((*tensor.shape, n_labels))
    return one_hot


def get_batches(arr, batch_size, seq_length):
    """
    we create a generator that returns tensor batches of size batch_size * seq_length from an array
    Arguments:
    arr: the array the want to make batches from
    batch_size: the number of sequences per batch
    eq_length: the number of encoded characters in a sequence
    """

    # we calculate the total amount of characters in a batch size
    total_batch_size = batch_size * seq_length
    # we then calculate the number of batches in the array (integer division)
    n_batches = len(arr) // total_batch_size

    # we drop the last few characters in order to create full batches
    arr = arr[:n_batches * total_batch_size]
    # then reshape into batch size rows
    arr = arr.reshape((batch_size, -1))

    # we iterate over the batches using a window of size seq_length
    for n in range(0, arr.shape[1], seq_length):
        # the features
        x = arr[:, n:n + seq_length]
        # the targets, shifted by one
        # we first create a y array of zeros, similar to x
        y = np.zeros_like(x)
        # and then, we assign to each of the y elements the elements of x, shifted by 1
        y[:, :-1] = x[:, 1:]
        # the last element will be assigned the next element from the original array
        try:
            y[:, -1] = arr[:, n + seq_length]
        # if however, we reach the end,
        # the last element of y will be assigned the first element of the original array,
        # in a cyclical way
        except IndexError:
            y[:, -1] = arr[:, 0]
        yield x, y
