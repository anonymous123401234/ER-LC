# Quick Verification

We have provided the final results and all intermediate outcomes in JSON or PKL formats for verification. To save storage space, all results have been compressed into ZIP format.

- **Step 1: Candidate Explanations Generation**

  - Access raw generated explanations by OpenAI at `candidate_reasons/openai_reasons_cache/`.
  - Constructed database is located at `reasons_cf_datasets/`.
- **Step 2: Debias Explanation Selection**

  - Selected explanations are available at `reasons_cf_datasets/dataset_name/user2reasons.json`.
- **Step 3: Recommendation Prompt and Results**

  - Recommendation prompt is provided at `recommendation_prompt4LLM/with_reasons_outputs`.
  - Recommendation results can be found at `recommendation_prompt4LLM/with_reasons_results`.

# Getting Started

Please note, the complete implementation code is not included here. Necessary modifications to the Python RecBole package (sections with ~~strikethrough~~ below) are also omitted. The full code will be released after our paper is published.

## Required Packages

```markdown
cuml==24.4.0
torch==2.3.0
recbole==1.2.0
scipy==1.12.0
scikit-learn==1.4.1
sentence-transformers==2.7.0
transformers==4.40.2
vllm==0.4.2
```

## Modifications of Recbole Package

~~Please copy the scripts from the `recbole/` directory to your Python RecBole package directory to replace the existing files.~~

```markdown
recbole/
├── trainer/
│   └── trainer.py
├── data/
│   ├── dataloader/
│   │   └── general_dataloader.py
│   └── utils.py
├── sampler/
    └── sampler.py
```

## Datasets

We perform our experiments on the [Movielens](https://files.grouplens.org/datasets/movielens/ml-latest-small.zip) and [Amazon_CDs_and_Vinyl](https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/CDs_and_Vinyl_5.json.gz) datasets. Data is converted to the RecBole-specific format using their official script, available at [RecBole](https://github.com/RUCAIBox/RecSysDatasets). Processed data and our subset creation scripts are located in the `data/` folder.

### Note for Dataset Usage:

This setup is designed for the MovieLens datasets. For the Amazon_CDs_and_Vinyl dataset, adjust the dataset variable to `amazoncd` at the beginning of each script. Check paths to ensure they are absolute to avoid bugs related to relative addressing.

## Detailed Steps

### Step 1: Candidate Explanations Generation

Generate candidate explanations using the OpenAI API:

```bash
# get the prompts for explanations generation
python get_rs_reasons_prompt.py

# generate explanations with API
python openai.py
python correct_id.py

# construct the database of candidate explanations
python candidate_reasons/dbscan_embedding_openai.py
```

### Step 2: Debias Explanation Selection

```bash
# Calculate item popularity
python calculate_popularity.py

# Train the debias model
bash rs_models/scripts/CF_model_train_movielens_pop.sh
bash rs_models/scripts/CF_model_train_amazoncd_pop.sh

# Retrieve selected explanations
python CF_model_test_debias.py
```

### Step 3: LLM-based Recommendation

```bash
# Generate recommendation prompts
python recommendation_prompt_generation.py

# Rank with LLMs
python recommendation_prompt4LLM/openai_recommendation.py

# Convert format for evaluation
python recommendation_prompt4LLM/combinations.py

# Evaluate
python llm_evaluation.py
```

## Baselines and Local LLM Usage

We also provide implementation details for some baseline models and guidance on using local LLMs with adjusted hyperparameters for proper output generation.

#TODO Please note that the setup details for the baseline and local LLM usage have not yet been finalized. Adjustments to the hyperparameters may be necessary, and some bugs may still need to be addressed.

```bash
# Step1 
# get the prompts for explanations generation 
python get_rs_reasons.py python correct_id.py 
# construct the database of candidate explanations 
python candidate_reasons/dbscan_embedding.py 


# Step2 is same
# Calculate item popularity
python calculate_popularity.py
# Train the debias model
bash rs_models/scripts/CF_model_train_movielens_pop.sh
bash rs_models/scripts/CF_model_train_amazoncd_pop.sh
# Retrieve selected explanations
python CF_model_test_debias.py


# Step 3 
python LLMRS_prompt.py
```

## Reference

- RecBole: [Official GitHub Repository](https://github.com/RUCAIBox/RecBole)
