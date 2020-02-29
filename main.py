# Part 1: importing

import torch
from torch.utils.data import DataLoader
from models import CharRNN
from datasets import CharacterDataset
from helper import validation_split, count_parameters, sample
from training import train

# we always check whether cuda is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    # Part 2: data processing

    # load the text data
    # in this case we will use Homer's Odyssey english translation
    with open('data/Odyssey.txt', 'r') as f:
        text = f.read()

    # we calculate our character vocabulary and the number of characters in it (number of classes)
    vocabulary = tuple(set(text))
    num_chars = len(vocabulary)

    # we also make our character to integer translation dictionaries
    int2char = dict(enumerate(vocabulary))
    char2int = {ch: i for i, ch in int2char.items()}

    # split our data into training and validation sets
    # in the cases of RNNs, shuffling our data would be catastrophic
    training_text, validation_text = validation_split(text)

    # declare the batch size in sequences and the character length of each sequence
    batch_size = 10
    seq_length = 50

    # prepare the dataset from our data
    train_data = CharacterDataset(training_text, vocabulary,
                                  batch_size=batch_size, seq_length=seq_length, device=device)
    validation_data = CharacterDataset(validation_text, vocabulary,
                                       batch_size=batch_size, seq_length=seq_length, device=device)

    # and make our data loaders
    # batch size is exactly 1 character by default, which is exactly what we need
    train_loader = DataLoader(train_data)
    validation_loader = DataLoader(validation_data)

    # Part 3: modelling
    # we create our model
    model = CharRNN(num_chars).to(device)
    # and the initial hidden state (a tensor of zeros)
    initial_state = model.init_hidden(batch_size, device)

    # we evaluate the capability of our model
    # a character to parameter ratio approaching 1 is optimal
    # too many parameters and the model may overfit
    # too few and the model may underfit
    char_param_ratio = len(text) / count_parameters(model)
    print("Character to model parameter ratio: %f\n" % char_param_ratio)

    # Part 4: training
    train(model, initial_state, train_loader=train_loader, validation_loader=validation_loader, epochs=100)

    # Part 5: evaluation
    print(sample(model, char2int))
