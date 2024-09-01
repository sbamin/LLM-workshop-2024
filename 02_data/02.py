# %% [markdown]
# **LLM Workshop 2024 by Sebastian Raschka**
# 
# This code is based on *Build a Large Language Model (From Scratch)*, [https://github.com/rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)

# %% [markdown]
# <br>
# <br>
# <br>
# <br>
# 
# # 2) Understanding LLM Input Data

# %% [markdown]
# Packages that are being used in this notebook:

# %%
from importlib.metadata import version


print("torch version:", version("torch"))
print("tiktoken version:", version("tiktoken"))

# %% [markdown]
# - This notebook provides a brief overview of the data preparation and sampling procedures to get input data "ready" for an LLM
# - Understanding what the input data looks like is a great first step towards understanding how LLMs work

# %% [markdown]
# <img src="./figures/01.png" width="1000px">

# %% [markdown]
# <br>
# <br>
# <br>
# <br>
# 
# # 2.1 Tokenizing text

# %% [markdown]
# - In this section, we tokenize text, which means breaking text into smaller units, such as individual words and punctuation characters

# %% [markdown]
# <img src="figures/02.png" width="800px">

# %% [markdown]
# - Load raw text we want to work with
# - [The Verdict by Edith Wharton](https://en.wikisource.org/wiki/The_Verdict) is a public domain short story

# %%
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
    
print("Total number of character:", len(raw_text))
print(raw_text[:99])

# %% [markdown]
# - The goal is to tokenize and embed this text for an LLM
# - Let's develop a simple tokenizer based on some simple sample text that we can then later apply to the text above

# %% [markdown]
# <img src="figures/03.png" width="690px">

# %% [markdown]
# - The following regular expression will split on whitespaces and punctuation

# %%
import re

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item for item in preprocessed if item]
print(preprocessed[:38])

# %%
print("Number of tokens:", len(preprocessed))

# %% [markdown]
# <br>
# <br>
# <br>
# <br>
# 
# # 2.2 Converting tokens into token IDs

# %% [markdown]
# - Next, we convert the text tokens into token IDs that we can process via embedding layers later
# - For this we first need to build a vocabulary

# %% [markdown]
# <img src="figures/04.png" width="900px">

# %% [markdown]
# - The vocabulary contains the unique words in the input text

# %%
all_words = sorted(set(preprocessed))
vocab_size = len(all_words)

print(vocab_size)

# %%
vocab = {token:integer for integer,token in enumerate(all_words)}

# %% [markdown]
# - Below are the first 50 entries in this vocabulary:

# %%
for i, item in enumerate(vocab.items()):
    print(item)
    if i >= 50:
        break

# %% [markdown]
# - Below, we illustrate the tokenization of a short sample text using a small vocabulary:

# %% [markdown]
# <img src="figures/05.png" width="800px">

# %% [markdown]
# - Let's now put it all together into a tokenizer class

# %%
class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}
    
    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
        
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text

# %% [markdown]
# - The `encode` function turns text into token IDs
# - The `decode` function turns token IDs back into text

# %% [markdown]
# <img src="figures/06.png" width="800px">

# %% [markdown]
# - We can use the tokenizer to encode (that is, tokenize) texts into integers
# - These integers can then be embedded (later) as input of/for the LLM

# %%
tokenizer = SimpleTokenizerV1(vocab)

text = """"It's the last he painted, you know," 
           Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
print(ids)

# %% [markdown]
# - We can decode the integers back into text

# %%
tokenizer.decode(ids)

# %%
tokenizer.decode(tokenizer.encode(text))

# %% [markdown]
# <br>
# <br>
# <br>
# <br>
# 
# # 2.3 BytePair encoding

# %% [markdown]
# - GPT-2 used BytePair encoding (BPE) as its tokenizer
# - it allows the model to break down words that aren't in its predefined vocabulary into smaller subword units or even individual characters, enabling it to handle out-of-vocabulary words
# - For instance, if GPT-2's vocabulary doesn't have the word "unfamiliarword," it might tokenize it as ["unfam", "iliar", "word"] or some other subword breakdown, depending on its trained BPE merges
# - The original BPE tokenizer can be found here: [https://github.com/openai/gpt-2/blob/master/src/encoder.py](https://github.com/openai/gpt-2/blob/master/src/encoder.py)
# - In this lecture, we are using the BPE tokenizer from OpenAI's open-source [tiktoken](https://github.com/openai/tiktoken) library, which implements its core algorithms in Rust to improve computational performance
# - (Based on an analysis [here](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch02/02_bonus_bytepair-encoder/compare-bpe-tiktoken.ipynb), I found that `tiktoken` is approx. 3x faster than the original tokenizer and 6x faster than an equivalent tokenizer in Hugging Face)

# %%
# pip install tiktoken

# %%
import importlib
import tiktoken

print("tiktoken version:", importlib.metadata.version("tiktoken"))

# %%
tokenizer = tiktoken.get_encoding("gpt2")

# %%
text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
     "of someunknownPlace."
)

integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})

print(integers)

# %%
strings = tokenizer.decode(integers)

print(strings)

# %% [markdown]
# - BPE tokenizers break down unknown words into subwords and individual characters:

# %% [markdown]
# <img src="figures/07.png" width="700px">

# %%
tokenizer.encode("Akwirw ier", allowed_special={"<|endoftext|>"})

# %% [markdown]
# <br>
# <br>
# <br>
# <br>
# 
# # 2.4 Data sampling with a sliding window

# %% [markdown]
# - Above, we took care of the tokenization (converting text into word tokens represented as token ID numbers)
# - Now, let's talk about how we create the data loading for LLMs
# - We train LLMs to generate one word at a time, so we want to prepare the training data accordingly where the next word in a sequence represents the target to predict

# %% [markdown]
# <img src="figures/08.png" width="800px">

# %% [markdown]
# - For this, we use a sliding window approach, changing the position by +1:
# 
# <img src="figures/09.png" width="900px">

# %% [markdown]
# - Note that in practice it's best to set the stride equal to the context length so that we don't have overlaps between the inputs (the targets are still shifted by +1 always)

# %% [markdown]
# <img src="figures/10.png" width="800px">

# %%
from supplementary import create_dataloader_v1


dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)

data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Inputs:\n", inputs)
print("\nTargets:\n", targets)

# %% [markdown]
# <br>
# <br>
# <br>
# <br>
# 
# # Exercise: Prepare your own favorite text dataset


