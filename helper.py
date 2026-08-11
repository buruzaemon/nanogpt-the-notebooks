import os
import torch

from pathlib import Path

PROJECT_ROOT = Path.cwd()

# download the tiny shakespeare dataset
input_file_path = os.path.join(PROJECT_ROOT, 'input.txt')
if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(input_file_path, 'w') as f:
        f.write(requests.get(data_url).text)

with open(input_file_path, 'r') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

# create mapping from chars to int
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# encode the entire text dataset and store into torch.Tensor
data = torch.tensor(encode(text), dtype=torch.long)

# split up the data into train and validation datasets
n = int(0.9*len(data))
training_data = data[:n]
validation_data = data[n:]


# ----------------------------------------------------
# CSS style definition to help with MathJAX formatting
#
from IPython.display import HTML, display

display(HTML("""
<style>
mjx-container[display="true"] {
    text-align: left !important;
    margin-left: 4em !important;
    margin-right: 0 !important;
}
</style>
"""))
