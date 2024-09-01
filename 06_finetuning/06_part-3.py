# %% [markdown]
# **LLM Workshop 2024 by Sebastian Raschka**

# %% [markdown]
# <br>
# <br>
# <br>
# <br>
# 
# # 6) Instruction finetuning (part 3; benchmark evaluation)

# %% [markdown]
# - In the previous notebook, we finetuned the LLM; in this notebook, we evaluate it using popular benchmark methods
# 
# - There are 3 main types of model evaluation
# 
#   1. MMLU-style Q&A
#   2. LLM-based automatic scoring
#   3. Human ratings by relative preference
#   
#   
# 

# %% [markdown]
# <img src="figures/10.png" width=800px>
# 
# <img src="figures/11.png" width=800px>

# %% [markdown]
# 
# <br>
# <br>
# <br>
# 
# 
# <img src="figures/13.png" width=800px>
# 
# 

# %% [markdown]
# <img src="figures/14.png" width=800px>
# 
# 

# %% [markdown]
# ## https://tatsu-lab.github.io/alpaca_eval/

# %% [markdown]
# <img src="figures/15.png" width=800px>
# 
# ## https://chat.lmsys.org

# %% [markdown]
# <br>
# <br>
# <br>
# <br>
# 
# # 6.2 Evaluation

# %% [markdown]
# - In this notebook, we do an MMLU-style evaluation in LitGPT, which is based on the [EleutherAI LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness)
# - There are hundreds if not thousands of benchmarks; using the command below, we filter for MMLU subsets, because running the evaluation on the whole MMLU dataset would take a very long time

# %% [markdown]
# - Let's say we are intrested in the `mmlu_philosophy` subset, se can evaluate the LLM on MMLU as follows

# %% [markdown]
# <br>
# <br>
# <br>
# <br>
# 
# # Exercise 3: Evaluate the finetuned LLM

# %% [markdown]
# <br>
# <br>
# <br>
# <br>
# 
# # Solution

# %%
!litgpt evaluate out/finetune/lora/final --tasks "mmlu_philosophy" --batch_size 4


