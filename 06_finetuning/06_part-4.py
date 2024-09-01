# %% [markdown]
# **LLM Workshop 2024 by Sebastian Raschka**
# 

# %% [markdown]
# <br>
# <br>
# <br>
# <br>
# 
# # 6) Instruction finetuning (part 4; evaluating instruction responses locally using a Llama 3 model)

# %% [markdown]
# - This notebook uses an 8 billion parameter Llama 3 model through LitGPT to evaluate responses of instruction finetuned LLMs based on a dataset in JSON format that includes the generated model responses, for example:
# 
# 
# 
# ```python
# {
#     "instruction": "What is the atomic number of helium?",
#     "input": "",
#     "output": "The atomic number of helium is 2.",               # <-- The target given in the test set
#     "response_before": "\nThe atomic number of helium is 3.0", # <-- Response by an LLM
#     "response_after": "\nThe atomic number of helium is 2."    # <-- Response by a 2nd LLM
# },
# ```
# 
# - The code doesn't require a GPU and runs on a laptop (it was tested on a M3 MacBook Air)

# %%
from importlib.metadata import version

pkgs = ["tqdm",    # Progress bar
        ]

for p in pkgs:
    print(f"{p} version: {version(p)}")

# %% [markdown]
# <br>
# <br>
# <br>
# <br>
# 
# ## 6.1 Load JSON Entries

# %% [markdown]
# - Now, let's get to the data evaluation part
# - Here, we assume that we saved the test dataset and the model responses as a JSON file that we can load as follows:

# %%
import json

json_file = "test_response_before_after.json"

with open(json_file, "r") as file:
    json_data = json.load(file)

print("Number of entries:", len(json_data))

# %% [markdown]
# - The structure of this file is as follows, where we have the given response in the test dataset (`'output'`) and responses by two different models (`'response_before'` and `'response_after'`):

# %%
json_data[0]

# %% [markdown]
# - Below is a small utility function that formats the input for visualization purposes later:

# %%
def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. Write a response that "
        f"appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    instruction_text + input_text

    return instruction_text + input_text

print(format_input(json_data[0])) # input

# %%
json_data[0]["output"]

# %%
json_data[0]["response_before"]

# %% [markdown]
# - Now, let's try LitGPT to compare the model responses (we only evaluate the first 5 responses for a visual comparison):

# %%
from litgpt import LLM

llm = LLM.load("meta-llama/Meta-Llama-3-8B-Instruct")

# %%
from tqdm import tqdm


def generate_model_scores(json_data, json_key):
    scores = []
    for entry in tqdm(json_data, desc="Scoring entries"):
        prompt = (
            f"Given the input `{format_input(entry)}` "
            f"and correct output `{entry['output']}`, "
            f"score the model response `{entry[json_key]}`"
            f" on a scale from 0 to 100, where 100 is the best score. "
            f"Respond with the integer number only."
        )
        score = llm.generate(prompt, max_new_tokens=50)
        try:
            scores.append(int(score))
        except ValueError:
            continue

    return scores

# %% [markdown]
# <br>
# <br>
# <br>
# <br>
# 
# # Exercise: Evaluate the LLMs
# 
# - Now using the `generate_model_scores` function above, evaluate the finetuned (`response_before`) and non-finetuned model (`response_after`)
# - Apply this evaluation to the whole dataset and compute the average score of each model

# %% [markdown]
# <br>
# <br>
# <br>
# <br>
# 
# # Exercise: Evaluate the LLMs
# 
# - Now using the `generate_model_scores` function above, evaluate the finetuned (`response_before`) and non-finetuned model (`response_after`)
# - Apply this evaluation to the whole dataset and compute the average score of each model

# %% [markdown]
# <br>
# <br>
# <br>
# <br>
# 
# # Solution

# %%
for model in ("response_before", "response_after"):

    scores = generate_model_scores(json_data, model)
    print(f"\n{model}")
    print(f"Number of scores: {len(scores)} of {len(json_data)}")
    print(f"Average score: {sum(scores)/len(scores):.2f}\n")


