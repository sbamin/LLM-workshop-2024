# %% [markdown]
# **LLM Workshop 2024 by Sebastian Raschka**
# 
# This code is based on *Build a Large Language Model (From Scratch)*, [https://github.com/rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)
# 
# - Instruction finetuning from scratch: [ch07.ipynb](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch07/01_main-chapter-code/ch07.ipynb)

# %% [markdown]
# <br>
# <br>
# <br>
# <br>
# 
# # 6) Instruction finetuning (part 1; intro)

# %% [markdown]
# <img src="figures/01.png" width=1000px>

# %% [markdown]
# <br>
# <br>
# <br>
# <br>
# 
# # 6.1 Introduction to instruction finetuning

# %% [markdown]
# - We saw that pretraining an LLM involves a training procedure where it learns to generate one word at a time
# - Hence, a pretrained LLM is good at text completion, but it is not good at following instructions
# - In this last part of the workshop, we teach the LLM to follow instructions better

# %% [markdown]
# <img src="figures/02.png" width=800px>

# %% [markdown]
# <br>
# <br>
# <br>
# <br>
# 
# # 6.2 Preparing a dataset for supervised instruction finetuning

# %% [markdown]
# - We will work with a simple instruction dataset I prepared for this

# %%
import json


file_path = "LLM-workshop-2024/06_finetuning/instruction-data.json"

with open(file_path, "r") as file:
    data = json.load(file)
print("Number of entries:", len(data))

# %% [markdown]
# - Each item in the `data` list we loaded from the JSON file above is a dictionary in the following form

# %%
print("Example entry:\n", data[50])

# %% [markdown]
# - Note that the `'input'` field can be empty:

# %%
print("Another example entry:\n", data[999])

# %% [markdown]
# - Instruction finetuning is often referred to as "supervised instruction finetuning" because it involves training a model on a dataset where the input-output pairs are explicitly provided
# - There are different ways to format the entries as inputs to the LLM; the figure below illustrates two example formats that were used for training the Alpaca (https://crfm.stanford.edu/2023/03/13/alpaca.html) and Phi-3 (https://arxiv.org/abs/2404.14219) LLMs, respectively

# %% [markdown]
# <img src="figures/03.png" width=900px>

# %% [markdown]
# - Suppose we use Alpaca-style prompt formatting, which was the original prompt template for instruction finetuning
# - Shown below is how we format the input that we would pass as input to the LLM

# %%
def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

    return instruction_text + input_text

# %% [markdown]
# - A formatted response with input field looks like as shown below

# %%
model_input = format_input(data[50])
desired_response = f"\n\n### Response:\n{data[50]['output']}"

print(model_input + desired_response)

# %% [markdown]
# - Below is a formatted response without an input field

# %%
model_input = format_input(data[999])
desired_response = f"\n\n### Response:\n{data[999]['output']}"

print(model_input + desired_response)

# %% [markdown]
# - Tokenized, this looks like as follows
# 
# <img src="figures/04.png" width=1000px>

# %% [markdown]
# - To make it work with batches, we add "padding" tokens 

# %% [markdown]
# - Tokenized, this looks like as follows
# 
# <img src="figures/05.png" width=1000px>

# %% [markdown]
# - Above, only the inputs are shown for simplicity; however, similar to pretraining, the target tokens are shifted by 1 position:

# %% [markdown]
# - Tokenized, this looks like as follows
# 
# <img src="figures/06.png" width=700px>

# %% [markdown]
# - In addition, it is also common to mask the target text
# - By default, PyTorch has the `cross_entropy(..., ignore_index=-100)` setting to ignore examples corresponding to the label -100
# - Using this -100 `ignore_index`, we can ignore the additional end-of-text (padding) tokens in the batches that we used to pad the training examples to equal length
# - However, we don't want to ignore the first instance of the end-of-text (padding) token (50256) because it can help signal to the LLM when the response is complete

# %% [markdown]
# - Tokenized, this looks like as follows
# 
# <img src="figures/07.png" width=1000px>


