# ###################### 标签生成 ###########################

# import json
# import requests
# from time import sleep
# import numpy as np
# from sklearn.metrics import precision_recall_fscore_support, accuracy_score, jaccard_score

# # ----------------------------
# # 1. 设置 DeepSeek‑r1 API 参数
# # ----------------------------
# API_ENDPOINT = 'https://xiaoai.plus/v1/chat/completions'  # API 端点
# API_KEY = 'sk-jjo3hoJbwFoVobd6ubk5Ct5GKjST0REf8NKiS4IoTwNae61Y'  # 请替换为你的实际 API key

# # ----------------------------
# # 2. 定义实体标签、定义及示例
# # ----------------------------
# entity_definitions = {
#     "Theory": "Theoretical framework, paradigm or school of thought that characterizes the research. Sample: DIKW model, technology acceptance model, constructivism, social cognitive theory",
#     "Data_Collection": "Means used to collect data for the study. Sample: Questionnaires, interview, web collecting, observations, Delphi, Transaction analysis, focus group",
#     "Data_Analysis": "Methods used to process and analyze the data. Sample: regression analysis, structural equation model (SEM), content analysis, co-word analysis, machine learning, network analysis",
#     "Dataset": "Referring to the specific data sources used (data objects, platforms, databases, etc.). Sample: Weibo, CNKI, WOS, Scopus, Facebook, Twitter",
#     "Indicator": "Nomenclature of specific indicators to measure the research object or variable. Sample: user satisfaction, information utilization, click-through rate, response time, citation frequency, F-measure",
#     "Tool": "Tools or software used to perform analysis, coding, modeling. Sample: SPSS, NVivo, Gephi, Python, R, Stata",
#     "Other": "Terms related to the research methodology but difficult to categorize. Sample: labeling system, indicator system"
# }

# def build_entity_prompt(entity_definitions):
#     """构造实体识别任务的提示文本，包含实体类别、定义和示例"""
#     return "\n".join(f"{et}: {desc}" for et, desc in entity_definitions.items())

# entity_prompt = build_entity_prompt(entity_definitions)

# # 定义允许的实体标签集合（全部保留，不过滤 "Question"）
# allowed_labels = {"Theory", "Data_Collection", "Data_Analysis", "Dataset", "Indicator", "Tool", "Other"}
# # 定义标签映射，如需将 "Data_Collection" 输出为 "Data_collection"，其它保持不变
# label_mapping = {
#     "Data_Collection": "Data_collection",
#     "Theory": "Theory",
#     "Data_Analysis": "Data_Analysis",
#     "Dataset": "Dataset",
#     "Indicator": "Indicator",
#     "Tool": "Tool",
#     "Other": "Other"
# }

# # ----------------------------
# # 3. 定义调用 DeepSeek‑r1 API 的函数（实体识别），加入少量学习示例
# # ----------------------------
# few_shot_examples = '''
# Example 1:
# Abstract: "Purpose This research aims to investigate the effects of innovation types (exploratory innovation vs. exploitative innovation) on users' psychological perceptions..."
# Entities: [{"id":1,"label":"Theory","start_offset":381,"end_offset":406}, {"id":2,"label":"Data_Analysis","start_offset":544,"end_offset":572}, {"id":3,"label":"Data_Analysis","start_offset":577,"end_offset":594}]

# Example 2:
# Abstract: "Scientists' evaluations are commonly made by h-index calculated from citation levels of published papers..."
# Entities: [{"id":4,"label":"Indicator","start_offset":1547,"end_offset":1555}, {"id":5,"label":"Indicator","start_offset":447,"end_offset":467}, {"id":6,"label":"Data_Analysis","start_offset":750,"end_offset":783}, {"id":7,"label":"Data_Analysis","start_offset":1600,"end_offset":1618}]
# '''

# def clean_json_output(output):
#     """
#     尝试修正模型输出的 JSON 字符串：
#       - 去除 Markdown 代码块标记，并补全不完整的 JSON
#     """
#     if output.startswith("```"):
#         lines = output.splitlines()
#         if lines[0].startswith("```"):
#             lines = lines[1:]
#         if lines and lines[-1].strip().startswith("```"):
#             lines = lines[:-1]
#         output = "\n".join(lines).strip()
#     if not output.endswith("]"):
#         pos = output.rfind("}")
#         if pos != -1:
#             output = output[:pos+1] + "]"
#     return output

# def extract_entities_gpt(text, timeout=60, max_retries=2):
#     retries = 0
#     while retries <= max_retries:
#         try:
#             messages = [
#                 {"role": "system", "content": "You are DeepSeek-v3, an expert specialized in academic named entity recognition."},
#                 {"role": "assistant", "content": few_shot_examples},
#                 {"role": "user", "content": (
#                     "Extract all entities from the following abstract. "
#                     "Return the results as a JSON array, where each element is an object with keys: "
#                     "'entity', 'type', 'start_offset', 'end_offset'. "
#                     "Only extract entities strictly belonging to the following categories:\n\n" +
#                     entity_prompt +
#                     "\n\nAbstract:\n" + text
#                 )}
#             ]
#             headers = {
#                 "Authorization": f"Bearer {API_KEY}",
#                 "Content-Type": "application/json"
#             }
#             payload = {"model": "deepseek-v3", "messages": messages, "max_tokens": 300}
#             response = requests.post(API_ENDPOINT, headers=headers, json=payload, timeout=timeout)
#             response.raise_for_status()
#             response_data = response.json()
#             output = response_data["choices"][0]["message"]["content"].strip()
#             if not output:
#                 print("Empty response received from DeepSeek-v3.")
#             output = clean_json_output(output)
#             try:
#                 entities = json.loads(output)
#                 if isinstance(entities, list):
#                     return entities
#                 else:
#                     print("DeepSeek-v3 output is not a list, got type:", type(entities))
#                     return []
#             except Exception as e:
#                 print("Error parsing DeepSeek-v3 output as JSON:", e)
#                 print("Raw DeepSeek-v3 output:", output)
#                 return []
#         except requests.exceptions.Timeout:
#             print("Request timed out, retrying...")
#             retries += 1
#             sleep(1)
#         except Exception as e:
#             print("Error calling DeepSeek-v3 API:", e)
#             return []
#     return []

# # ----------------------------
# # 4. 读取测试集数据 & 调用 GPT API生成实体识别结果
# # ----------------------------
# # 测试集文件路径（训练集处理类似，数据已提前划分好）
# test_input_file = "/workspace/Innovation_method_202503/input_data/3_method_test.jsonl"
# results = []

# with open(test_input_file, 'r', encoding='utf-8') as f:
#     data_samples = [json.loads(line) for line in f]

# for sample in data_samples:
#     paper_id = sample.get("Paperid")
#     text = sample["text"]
    
#     print(f"Processing PaperID: {paper_id} ...")
#     gpt_entities_raw = extract_entities_gpt(text)
#     # 将所有 DeepSeek-v3 返回的实体转换，进行标签映射
#     filtered_entities = []
#     entity_id = 1
#     for ent in gpt_entities_raw:
#         filtered_entities.append({
#             "id": entity_id,
#             "label": label_mapping.get(ent.get("type", ""), ent.get("type", "")),
#             "start_offset": ent.get("start_offset"),
#             "end_offset": ent.get("end_offset")
#         })
#         entity_id += 1

#     result_obj = {
#         "Paperid": paper_id,
#         "text": text,
#         "Publication Year": sample.get("Publication Year"),
#         "Source Title": sample.get("Source Title", ""),
#         "DOI": sample.get("DOI", ""),
#         "entities": filtered_entities
#     }
#     results.append(result_obj)
    
#     print(f"PaperID {paper_id} processed. DeepSeek-v3 Entities: {filtered_entities}")
#     print("-" * 50)
#     sleep(3)

# # 保存测试集输出结果为 JSON Lines 文件
# output_test_file = "/workspace/Innovation_method_202503/output_data/deepseekv3_method_test_output.jsonl"
# with open(output_test_file, 'w', encoding='utf-8') as f_out:
#     for item in results:
#         f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
# print(f"Test output saved to {output_test_file}")

###################### 评估模型 ###########################
import json
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def fuzzy_match(e1, e2, threshold=0.5):
    """
    判断两个实体是否模糊匹配。
    e1, e2 均为元组 (label, start_offset, end_offset)。
    要求标签相同，并且两个实体的交集/并集 (IoU) 不小于 threshold。
    """
    if e1[0] != e2[0]:
        return False
    start1, end1 = e1[1], e1[2]
    start2, end2 = e2[1], e2[2]
    overlap = max(0, min(end1, end2) - max(start1, start2))
    union = max(end1, end2) - min(start1, start2)
    if union == 0:
        return False
    return (overlap / union) >= threshold

# ----------------------------
# 1. 加载原始测试集（真实标签）和模型输出文件（预测结果）
# ----------------------------
# 原始测试集文件，包含真实实体标签
input_file = "/workspace/Innovation_method_202503/input_data/3_method_test.jsonl"
# 模型输出文件，包含预测实体结果
output_file = "/workspace/Innovation_method_202503/output_data/deepseekv3_method_test_output.jsonl"

true_records = []
with open(input_file, 'r', encoding='utf-8') as fin:
    for line in fin:
        true_records.append(json.loads(line))

pred_records = []
with open(output_file, 'r', encoding='utf-8') as fin:
    for line in fin:
        pred_records.append(json.loads(line))

# ----------------------------
# 2. 构造实体集合（每个实体用 (label, start_offset, end_offset) 表示）
# ----------------------------
# 假设两边记录可通过 "Paperid" 进行匹配
true_dict = {rec.get("Paperid"): rec for rec in true_records}
pred_dict = {rec.get("Paperid"): rec for rec in pred_records}

all_true_entities_total = []
all_pred_entities_total = []

for paperid, true_rec in true_dict.items():
    true_entities = true_rec.get("entities", [])
    pred_entities = pred_dict.get(paperid, {}).get("entities", [])
    
    # 将实体转换为元组形式
    true_set = set((ent.get("label"), ent.get("start_offset"), ent.get("end_offset")) for ent in true_entities)
    pred_set = set((ent.get("label"), ent.get("start_offset"), ent.get("end_offset")) for ent in pred_entities)
    
    all_true_entities_total.append(true_set)
    all_pred_entities_total.append(pred_set)

# ----------------------------
# 3. 评估
# ----------------------------
TP_total = 0
FP_total = 0
FN_total = 0
sample_exact_matches = 0
num_samples = len(all_true_entities_total)

for true_set, pred_set in zip(all_true_entities_total, all_pred_entities_total):
    true_list = list(true_set)
    pred_list = list(pred_set)
    matched_true = set()
    matched_pred = set()
    # 如果两边均为空，则认为完全匹配
    if len(true_list) == 0 and len(pred_list) == 0:
        sample_exact_matches += 1
    else:
        # 对每个预测实体尝试匹配一个真实实体（模糊匹配）
        for i, pred in enumerate(pred_list):
            for j, true in enumerate(true_list):
                if j not in matched_true and fuzzy_match(pred, true, threshold=0.5):
                    matched_true.add(j)
                    matched_pred.add(i)
                    break
        TP = len(matched_true)
        FP = len(pred_list) - len(matched_pred)
        FN = len(true_list) - len(matched_true)
        TP_total += TP
        FP_total += FP
        FN_total += FN
        if len(true_list) > 0 and len(true_list) == len(pred_list) == TP:
            sample_exact_matches += 1

entity_precision = TP_total / (TP_total + FP_total) if (TP_total + FP_total) > 0 else 0.0
entity_recall = TP_total / (TP_total + FN_total) if (TP_total + FN_total) > 0 else 0.0
entity_f1 = 2 * entity_precision * entity_recall / (entity_precision + entity_recall) if (entity_precision + entity_recall) > 0 else 0.0
entity_exact_accuracy = sample_exact_matches / num_samples if num_samples > 0 else 0.0

global_true = set.union(*all_true_entities_total) if all_true_entities_total else set()
global_pred = set.union(*all_pred_entities_total) if all_pred_entities_total else set()
global_jaccard = len(global_true.intersection(global_pred)) / (len(global_true.union(global_pred)) + 1e-8)

print("\n--- Fuzzy Entity Recognition Evaluation ---")
print(f"Entity Precision: {entity_precision:.4f}")
print(f"Entity Recall:    {entity_recall:.4f}")
print(f"Entity F1:        {entity_f1:.4f}")
print(f"Entity Exact Accuracy: {entity_exact_accuracy:.4f}")
print(f"Entity Jaccard:   {global_jaccard:.4f}")