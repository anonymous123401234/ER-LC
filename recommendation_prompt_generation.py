import json
import pandas as pd
import os
import pickle
import re


dataset = "movielens"
topk_reasons = [5]
output_path = "recommendation_prompt4LLM/with_reasons_output"

feature_type = "only_name"
if dataset == "amazoncd":
    reasons_predict_path = "reasons_cf_datasets/cluster_results_MiniLM_UMAP20openai_amazoncd"
    raw_prompt_file = f"/recommendation_prompt4LLM/Amazon_CDs_and_Vinyl_small_{feature_type}_prompt.json"
else:
    reasons_predict_path = "reasons_cf_datasets/cluster_results_MiniLM_UMAP20_openai_movielens"
    raw_prompt_file = f"/recommendation_prompt4LLM/ml-latest-small_{feature_type}_prompt.json"
all_file_list = os.listdir(reasons_predict_path)
all_file_list = [i for i in all_file_list if "user2reasons" in i]
print(all_file_list)

with open(raw_prompt_file, 'r', encoding='utf8') as f:
    raw_data = json.load(f)
raw_data

with open(reasons_predict_path + "/" + "user_item2reasonsid_text.pkl", 'rb') as f:
    user_reasonid2text = pickle.load(f)
reasonsid2text = {}
for user_id, user_reasons in user_reasonid2text['user'].items():
    for per_reasons_id in user_reasons:
        if per_reasons_id not in reasonsid2text:
            reasonsid2text[per_reasons_id] = user_reasons[per_reasons_id]
        reasonsid2text[per_reasons_id] = reasonsid2text[per_reasons_id] + \
            user_reasons[per_reasons_id]
reasonsid2text


for topk in topk_reasons:
    for per_file in all_file_list:
        if "_100" in per_file:
            continue
        with open(reasons_predict_path+"/"+per_file, 'r', encoding='utf8') as f:
            userid2predicted_reasons = json.load(f)
        withreasons_dict = {}
        for user_id, value in raw_data.items():
            if user_id not in userid2predicted_reasons:
                print(user_id)
                continue
            predicted_reasons_id = userid2predicted_reasons[user_id][0][:topk]
            predicted_reasons_text = []
            for i in predicted_reasons_id:
                if user_id in user_reasonid2text['user'] and i in user_reasonid2text['user'][user_id]:
                    predicted_reasons_text.append(
                        user_reasonid2text['user'][user_id][i][0])
                else:
                    if int(i) == -1:
                        continue
                    predicted_reasons_text.append(reasonsid2text[int(i)][0])
            predicted_reasons_text = [
                re.sub(r'^\d+\.\s*', '', i) for i in predicted_reasons_text]
            if dataset == "amazoncd":
                raw_prompt = value[3]
                raw_prompt = raw_prompt.split("Please rank the candidate CDs")
                raw_prompt = raw_prompt[0] + "and the reasons for potentially choosing to watch, listed as: " + \
                    ",\n".join(predicted_reasons_text) + \
                    "\nPlease rank the candidate CDs" + raw_prompt[1]
            elif dataset == "movielens":
                raw_prompt = value[3]
                raw_prompt = raw_prompt.split(
                    "please rank the candidate movies")
                raw_prompt = raw_prompt[0] + "and the reasons for potentially choosing to watch, listed as: " + \
                    ",\n".join(predicted_reasons_text) + \
                    "\nPlease rank the candidate movies" + raw_prompt[1]
            else:
                raise NotImplementedError
            withreasons_dict[user_id] = [
                value[0], value[1], value[2], raw_prompt]
        with open(output_path + "/" + f"{dataset}_" + per_file[:-5] + f"_{feature_type}_top{topk}.json", 'w', encoding='utf8') as f:
            json.dump(withreasons_dict, f)
        print("Down")



