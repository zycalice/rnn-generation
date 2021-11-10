import torch
from torch import nn
import numpy as np

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# This function takes in the model and character as arguments and returns the next character prediction and hidden state
def predict(model, dp, character, random=True):
    character = dp.make_one_hot(character)
    character = torch.from_numpy(character).float()
    character.to(DEVICE)

    out, hidden = model(character.view(1, 25, 81))

    # Taking the class with the highest probability score from the output (greedy)
    prob = nn.functional.softmax(out[-1], dim=0).data
    char_ind = torch.max(prob, dim=0)[1].item()

    if random:
        char_ind = np.searchsorted(
            np.cumsum(prob),
            np.random.rand()
        ).item()

    return dp.num2char[char_ind], hidden


# This function takes the desired output length and input characters as arguments, returning the produced sentence
def sample(model, out_len, dp):
    model.eval()  # eval mode
    # First off, run through the starting characters
    chars = dp.get_random_chunk(24)
    initial = chars
    pred = chars
    size = out_len - len(chars)
    # Now pass in the previous characters and get a new one
    for ii in range(size):
        char, h = predict(model, dp, chars)
        chars = chars[1:] + char
        pred += char

    return initial, pred
