import numpy as np


# splits string data into training and validation sets
def validation_split(data, val_frac=0.1):
    val_idx = int(len(data) * (1 - val_frac))
    return data[:val_idx], data[val_idx:]


# for n labels and a 2-D arr of integers,
# return a one-hot 2-D numpy array representation
def one_hot_encode(arr, n_labels):
    # create an initial one-hot array filled with zeros with the size of the array and the number of labels
    one_hot = np.zeros((arr.size, n_labels), dtype=np.float32)

    # fill in the appropriate elements with ones
    # arr as we already mentioned is 2-D array so we flatten it
    one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1

    # finally, we reshape to get back to the original array
    one_hot = one_hot.reshape((*arr.shape, n_labels))

    return one_hot


# we create a generator that returns tensor batches of size batch_size * seq_length from an array
# Arguments:
# arr: the array the want to make batches from
# batch_size: the number of sequences per batch
# seq_length: the number of encoded characters in a sequence
def get_batches(arr, batch_size, seq_length):
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
