import re
import json
from recbole.utils import init_seed, init_logger
from recbole.config import Config
from logging import getLogger
from load_datasets import load_recbole_datasets



dataset_name = "movielens"
# dataset_name = "amazon_cd"


if dataset_name == "amazon_cd":
    config = Config(model='BERT4Rec', dataset='Amazon_CDs_and_Vinyl_small', config_file_list=[
        "config/sequential_ml_amazon.yaml", "config/LLM.yaml"])
    config["user_inter_num_interval"] = None
    config["item_inter_num_interval"] = None
elif dataset_name == "movielens":
    config = Config(model='BERT4Rec', dataset='ml-latest-small', config_file_list=[
        "config/sequential_ml.yaml", "config/LLM.yaml"])
init_logger(config)
init_seed(config['seed'], config['reproducibility'])
logger = getLogger()
logger.info(config)

train_data, valid_data, test_data = load_recbole_datasets(logger, config)

model_userid2raw_userid = train_data._dataset.field2id_token['user_id']
model_itemid2raw_item_id = train_data._dataset.field2id_token['item_id']


if dataset_name == "amazon_cd":
    raw_get_reasons_file_name = "candidate_reasons/prompts_for_generating_explanations/get_reasons_for_LLMAPI_Amazon_CDs_and_Vinyl_small.json"
    # corrected_raw_get_reasons_file_name = "get_openai_reasons_amazoncd.json"
    raw_openai_reasons_file_name = "candidate_reasons/openai_reasons_cache/openai_reasons_amazoncd.txt"
    corrected_openai_reasons_file_name = "candidate_reasons/raw_generated_reasons/openai_reasons_amazoncd.json"

elif dataset_name == "movielens":
    raw_get_reasons_file_name = "candidate_reasons/prompts_for_generating_explanations/get_reasons_for_LLMAPI_ml-latest-small.json"
    # corrected_raw_get_reasons_file_name = "get_openai_reasons_movielens.json"
    raw_openai_reasons_file_name = "candidate_reasons/openai_reasons_cache/openai_reasons_ml100k.txt"
    corrected_openai_reasons_file_name = "candidate_reasons/raw_generated_reasons/openai_reasons_movielens.json"

with open(raw_get_reasons_file_name, 'r', encoding='utf8') as f:
    reasons = json.load(f)
corrected_reasons = {}
for key, value in reasons.items():
    corrected_reasons[model_userid2raw_userid[int(key)]] = value

# with open(corrected_output_path + corrected_raw_get_reasons_file_name, 'w', encoding='utf8') as f:
#     json.dump(corrected_reasons, f)

reasons_output = []
all_reasons = []
with open(raw_openai_reasons_file_name, 'r', encoding='utf8') as f:
    for line in f:
        openai_reasons = json.loads(line)
        model_user_id, model_item_id = openai_reasons["union_id"].split(
            ";")
        raw_user_id, raw_item_id = model_userid2raw_userid[int(
            model_user_id)], model_item_id
        pattern = re.compile(r'^\s*-\s+|^\s*\d+\.\s+')
        reasons_list = [pattern.sub('', i)
                        for i in openai_reasons['reasons'].split("\n")]
        reasons_output.append(
            {"user_id": raw_user_id, "item_id": raw_item_id, "reasons": reasons_list})
        for tmp in reasons_list:
            all_reasons.append(tmp)
with open(corrected_openai_reasons_file_name, 'w', encoding='utf8') as f:
    json.dump(reasons_output, f)