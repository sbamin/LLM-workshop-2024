# %% [markdown]
# **LLM Workshop 2024 by Sebastian Raschka**

# %% [markdown]
# <br>
# <br>
# <br>
# <br>
# 
# # 5) Loading pretrained weights (part 2; using LitGPT)

# %% [markdown]
# - Now, we are loading the weights using an open-source library called LitGPT
# - LitGPT is fundamentally similar to the LLM code we implemented previously, but it is much more sophisticated and supports more than 20 different LLMs (Mistral, Gemma, Llama, Phi, and more)
# 
# # ⚡ LitGPT
# 
# **20+ high-performance LLMs with recipes to pretrain, finetune, deploy at scale.**
# 
# <pre>
# ✅ From scratch implementations     ✅ No abstractions    ✅ Beginner friendly   
# ✅ Flash attention                  ✅ FSDP               ✅ LoRA, QLoRA, Adapter
# ✅ Reduce GPU memory (fp4/8/16/32)  ✅ 1-1000+ GPUs/TPUs  ✅ 20+ LLMs            
# </pre>
# 
# ## Basic usage:
# 
# ```
# # ligpt [action] [model]
# litgpt  download  meta-llama/Meta-Llama-3-8B-Instruct
# litgpt  chat      meta-llama/Meta-Llama-3-8B-Instruct
# litgpt  evaluate  meta-llama/Meta-Llama-3-8B-Instruct
# litgpt  finetune  meta-llama/Meta-Llama-3-8B-Instruct
# litgpt  pretrain  meta-llama/Meta-Llama-3-8B-Instruct
# litgpt  serve     meta-llama/Meta-Llama-3-8B-Instruct
# ```
# 
# 
# - You can learn more about LitGPT in the [corresponding GitHub repository](https://github.com/Lightning-AI/litgpt), that contains many tutorials, use cases, and examples
# 

# %%
# pip install litgpt

# %%
from importlib.metadata import version

pkgs = ["litgpt", 
        "torch",
       ]
for p in pkgs:
    print(f"{p} version: {version(p)}")

# %% [markdown]
# - First, let's see what LLMs are supported

# %%
!litgpt download list

# %% [markdown]
# - We can then download an LLM via the following command

# %%
!litgpt download microsoft/phi-2

# %% [markdown]
# - And there's also a Python API to use the model

# %%
from litgpt import LLM

llm = LLM.load("microsoft/phi-2")

llm.generate("What do Llamas eat?")

# %%
result = llm.generate("What do Llamas eat?", stream=True, max_new_tokens=200)
for e in result:
    print(e, end="", flush=True)

# %% [markdown]
# <br>
# <br>
# <br>
# <br>
# 
# # Exercise 2: Download an LLM

# %% [markdown]
# - Download and try out an LLM of your own choice (recommendation: 7B parameters or smaller)
# - We will finetune the LLM in the next notebook
# - You can also try out the `litgpt chat` command from the terminal


