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
# # 3) Coding an LLM architecture

# %%
from importlib.metadata import version


print("torch version:", version("torch"))
print("tiktoken version:", version("tiktoken"))

# %% [markdown]
# - In this notebook, we implement a GPT-like LLM architecture; the next notebook will focus on training this LLM

# %% [markdown]
# <img src="figures/01.png" width="1000px">

# %% [markdown]
# <br>
# <br>
# <br>
# <br>
# 
# # 3.1 Coding an LLM architecture

# %% [markdown]
# - Models like GPT, Gemma, Phi, Mistral, Llama etc. generate words sequentially and are based on the decoder part of the original transformer architecture
# - Therefore, these LLMs are often referred to as "decoder-like" LLMs
# - Compared to conventional deep learning models, LLMs are larger, mainly due to their vast number of parameters, not the amount of code
# - We'll see that many elements are repeated in an LLM's architecture

# %% [markdown]
# <img src="figures/02.png" width="700px">

# %% [markdown]
# - In the previous notebook, we used small embedding dimensions for token inputs and outputs for ease of illustration, ensuring they neatly fit on the screen
# - In this notebook, we consider embedding and model sizes akin to a small GPT-2 model
# - We'll specifically code the architecture of the smallest GPT-2 model (124 million parameters), as outlined in Radford et al.'s [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (note that the initial report lists it as 117M parameters, but this was later corrected in the model weight repository)
# 

# %% [markdown]
# 
# <img src="figures/03.png" width="1200px">
# 
# - The next notebook will show how to load pretrained weights into our implementation, which will be compatible with model sizes of 345, 762, and 1542 million parameters
# - Models like Llama and others are very similar to this model, since they are all based on the same core concepts
# 
# <img src="figures/04.png" width="1200px">

# %% [markdown]
# - Configuration details for the 124 million parameter GPT-2 model (GPT-2 "small") include:

# %%
GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 1024, # Context length
    "emb_dim": 768,         # Embedding dimension
    "n_heads": 12,          # Number of attention heads
    "n_layers": 12,         # Number of layers
    "drop_rate": 0.0,       # Dropout rate
    "qkv_bias": False       # Query-Key-Value bias
}

# %% [markdown]
# <br>
# <br>
# <br>
# <br>
# 
# 
# 
# # 3.2 Coding the GPT model

# %% [markdown]
# - We are almost there: now let's plug in the transformer block into the architecture we coded at the very beginning of this notebook so that we obtain a useable GPT architecture
# - Note that the transformer block is repeated multiple times; in the case of the smallest 124M GPT-2 model, we repeat it 12 times:

# %% [markdown]
# <img src="figures/07.png" width="800px">

# %% [markdown]
# - The corresponding code implementation, where `cfg["n_layers"] = 12`:

# %%
import torch.nn as nn
from supplementary import TransformerBlock, LayerNorm


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

# %% [markdown]
# - Using the configuration of the 124M parameter model, we can now instantiate this GPT model with random initial weights as follows:

# %%
import torch
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

batch = []

txt1 = "Every effort moves you"
txt2 = "Every day holds a"

batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)
print(batch)

# %%
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)

out = model(batch)
print("Input batch:\n", batch)
print("\nOutput shape:", out.shape)
print(out)

# %% [markdown]
# - We will train this model in the next notebook

# %% [markdown]
# <br>
# <br>
# <br>
# <br>
# 
# 
# 
# # 3.4 Generating text

# %% [markdown]
# - LLMs like the GPT model we implemented above are used to generate one word at a time

# %% [markdown]
# <img src="figures/08.png" width="600px">

# %% [markdown]
# - The following `generate_text_simple` function implements greedy decoding, which is a simple and fast method to generate text
# - In greedy decoding, at each step, the model chooses the word (or token) with the highest probability as its next output (the highest logit corresponds to the highest probability, so we technically wouldn't even have to compute the softmax function explicitly)
# - The figure below depicts how the GPT model, given an input context, generates the next word token

# %% [markdown]
# <img src="figures/09.png" width="900px">

# %%
def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx is (batch, n_tokens) array of indices in the current context
    for _ in range(max_new_tokens):
        
        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        idx_cond = idx[:, -context_size:]
        
        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond)
        
        # Focus only on the last time step
        # (batch, n_tokens, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]  

        # Apply softmax to get probabilities
        probas = torch.softmax(logits, dim=-1)  # (batch, vocab_size)

        # Get the idx of the vocab entry with the highest probability value
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx

# %% [markdown]
# - The `generate_text_simple` above implements an iterative process, where it creates one token at a time
# 
# <img src="figures/10.png" width="800px">

# %% [markdown]
# <br>
# <br>
# <br>
# <br>
# 
# 
# 
# # Exercise: Generate some text

# %% [markdown]
# 1. Use the `tokenizer.encode` method to prepare some input text
# 2. Then, convert this text into a pytprch tensor via (`torch.tensor`)
# 3. Add a batch dimension via `.unsqueeze(0)`
# 4. Use the `generate_text_simple` function to have the GPT generate some text based on your prepared input text
# 5. The output from step 4 will be token IDs, convert them back into text via the `tokenizer.decode` method

# %%
model.eval();  # disable dropout

# %% [markdown]
# <br>
# <br>
# <br>
# <br>
# 
# 
# 
# # Solution

# %%
start_context = "Hello, I am"

encoded = tokenizer.encode(start_context)
print("encoded:", encoded)

encoded_tensor = torch.tensor(encoded).unsqueeze(0)
print("encoded_tensor.shape:", encoded_tensor.shape)

# %%
out = generate_text_simple(
    model=model,
    idx=encoded_tensor, 
    max_new_tokens=6, 
    context_size=GPT_CONFIG_124M["context_length"]
)

print("Output:", out)
print("Output length:", len(out[0]))

# %% [markdown]
# - Remove batch dimension and convert back into text:

# %%
decoded_text = tokenizer.decode(out.squeeze(0).tolist())
print(decoded_text)

# %% [markdown]
# - Note that the model is untrained; hence the random output texts above
# - We will train the model in the next notebook


