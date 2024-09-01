# %% [markdown]
# **LLM Workshop 2024 by Sebastian Raschka**

# %% [markdown]
# <br>
# <br>
# <br>
# <br>
# 
# # 6) Instruction finetuning (part 2; finetuning)

# %% [markdown]
# - In this notebook, we get to the actual finetuning part
# - But first, let's briefly introduce a technique, called LoRA, that makes the finetuning more efficient
# - It's not required to use LoRA, but it can result in noticeable memory savings while still resulting in good modeling performance

# %% [markdown]
# <br>
# <br>
# <br>
# <br>
# 
# # 6.1 Introduction to LoRA

# %% [markdown]
# - Low-rank adaptation (LoRA) is a machine learning technique that modifies a pretrained model to better suit a specific, often smaller, dataset by adjusting only a small, low-rank subset of the model's parameters
# - This approach is important because it allows for efficient finetuning of large models on task-specific data, significantly reducing the computational cost and time required for finetuning

# %% [markdown]
# - Suppose we have a large weight matrix $W$ for a given layer
# - During backpropagation, we learn a $\Delta W$ matrix, which contains information on how much we want to update the original weights to minimize the loss function during training
# - In regular training and finetuning, the weight update is defined as follows:
# 
# $$W_{\text{updated}} = W + \Delta W$$
# 
# - The LoRA method proposed by [Hu et al.](https://arxiv.org/abs/2106.09685) offers a more efficient alternative to computing the weight updates $\Delta W$ by learning an approximation of it, $\Delta W \approx AB$.
# - In other words, in LoRA, we have the following, where $A$ and $B$ are two small weight matrices:
# 
# $$W_{\text{updated}} = W + AB$$
# 
# - The figure below illustrates these formulas for full finetuning and LoRA side by side

# %% [markdown]
# <img src="figures/08.png" width="1100px">

# %% [markdown]
# - If you paid close attention, the full finetuning and LoRA depictions in the figure above look slightly different from the formulas I have shown earlier
# - That's due to the distributive law of matrix multiplication: we don't have to add the weights with the updated weights but can keep them separate
# - For instance, if $x$ is the input data, then we can write the following for regular finetuning:
# 
# $$x (W+\Delta W) = x W + x \Delta W$$
# 
# - Similarly, we can write the following for LoRA:
# 
# $$x (W+A B) = x W + x A B$$
# 
# - The fact that we can keep the LoRA weight matrices separate makes LoRA especially attractive
# - In practice, this means that we don't have to modify the weights of the pretrained model at all, as we can apply the LoRA matrices on the fly
# - After setting up the dataset and loading the model, we will implement LoRA in the code to make these concepts less abstract

# %% [markdown]
# <img src="figures/09.png" width="800px">

# %% [markdown]
# <br>
# <br>
# <br>
# <br>
# 
# # 6.2 Creating training and test sets

# %% [markdown]
# - There's one more thing before we can start finetuning: creating the training and test subsets
# - We will use 85% of the data for training and the remaining 15% for testing

# %%
import json


file_path = "LLM-workshop-2024/06_finetuning/instruction-data.json"

with open(file_path, "r") as file:
    data = json.load(file)
print("Number of entries:", len(data))

# %%
train_portion = int(len(data) * 0.85)  # 85% for training
test_portion = int(len(data) * 0.15)    # 15% for testing

train_data = data[:train_portion]
test_data = data[train_portion:]

# %%
print("Training set length:", len(train_data))
print("Test set length:", len(test_data))

# %%
with open("train.json", "w") as json_file:
    json.dump(train_data, json_file, indent=4)
    
with open("test.json", "w") as json_file:
    json.dump(test_data, json_file, indent=4)

# %% [markdown]
# <br>
# <br>
# <br>
# <br>
# 
# # 6.3 Instruction finetuning

# %% [markdown]
# - Using LitGPT, we can finetune the model via `litgpt finetune model_dir`
# - However, here, we will use LoRA finetuning `litgpt finetune_lora model_dir` since it will be quicker and less resource intensive

# %%
!litgpt finetune_lora microsoft/phi-2 \
--data JSON \
--data.val_split_fraction 0.1 \
--data.json_path train.json \
--train.epochs 3 \
--train.log_interval 100

# %% [markdown]
# <br>
# <br>
# <br>
# <br>
# 
# # Exercise 1: Generate and save the test set model responses of the base model

# %% [markdown]
# - In this excercise, we are collecting the model responses on the test dataset so that we can evaluate them later
# 
# 
# - Starting with the original model before finetuning, load the model using the LitGPT Python API (`LLM.load` ...)
# - Then use the `LLM.generate` function to generate the responses for the test data
# - The following utility function will help you to format the test set entries as input text for the LLM

# %%
def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

    return instruction_text + input_text

print(format_input(test_data[0]))

# %%
from litgpt import LLM

llm = LLM.load("microsoft/phi-2")

# %%
from tqdm import tqdm

for i in tqdm(range(len(test_data))):
    response = llm.generate(test_data[i])
    test_data[i]["base_model"] = response

# %% [markdown]
# - Using this utility function, generate and save all the test set responses generated by the model and add them to the `test_set`
# - For example, if `test_data[0]` entry is as follows before:
#     
# ```
# {'instruction': 'Rewrite the sentence using a simile.',
#  'input': 'The car is very fast.',
#  'output': 'The car is as fast as lightning.'}
# ```
# 
# - Modify the `test_data` entry so that it contains the model response:
#     
# ```
# {'instruction': 'Rewrite the sentence using a simile.',
#  'input': 'The car is very fast.',
#  'output': 'The car is as fast as lightning.',
#  'base_model': 'The car is as fast as a cheetah sprinting across the savannah.'
# }
# ```
# 
# - Do this for all test set entries, and then save the modified `test_data` dictionary as `test_base_model.json`
# 

# %%
test_data[1]

# %% [markdown]
# <br>
# <br>
# <br>
# <br>
# 
# # Exercise 2: Generate and save the test set model responses of the finetuned model

# %% [markdown]
# - Repeat the steps from the previous exercise but this time collect the responses of the finetuned model
# - Save the resulting `test_data` dictionary as `test_base_and_finetuned_model.json`

# %% [markdown]
# <br>
# <br>
# <br>
# <br>
# 
# # Solution

# %%
from litgpt import LLM

del llm
llm2 = LLM.load("out/finetune/lora/final/")

# %%
from tqdm import tqdm

for i in tqdm(range(len(test_data))):
    response = llm2.generate(test_data[i])
    test_data[i]["finetuned_model"] = response


