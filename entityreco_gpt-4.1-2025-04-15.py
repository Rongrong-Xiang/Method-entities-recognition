import json
import requests
from time import sleep

# ----------------------------
# 1. GPT API设置
# ----------------------------
API_ENDPOINT = 'https://xiaoai.plus/v1/chat/completions'
API_KEY = 'sk-IIx0VwwYjt7hGhecJwiEEP0LIvtzioLfeyVCEEj0D7ZlZfO7' 

entity_definitions = {
    "Question": "The research question, objective, or main problem addressed in the study. Sample: What are the key factors influencing user satisfaction?; The aim of this study is to investigate the relationship between social media use and academic performance.",
    "Theory": "Theoretical framework, paradigm or school of thought that characterizes the research. Sample: DIKW model, technology acceptance model, constructivism, social cognitive theory",
    "Data_Collection": "Means used to collect data for the study. Sample: Questionnaires, interview, web collecting, observations, Delphi, Transaction analysis, focus group",
    "Data_Analysis": "Methods used to process and analyze the data. Sample: regression analysis, structural equation model (SEM), content analysis, co-word analysis, machine learning, network analysis,meta-analysis",
    "Dataset": "Referring to the specific data sources used (data objects, platforms, databases, etc.). Sample: Weibo, CNKI, WOS, Scopus, Facebook, Twitter",
    "Indicator": "Nomenclature of specific indicators to measure the research object or variable. Sample: user satisfaction, information utilization, click-through rate, response time, citation frequency, F-measure",
    "Tool": "Tools or software used to perform analysis, coding, modeling. Sample: SPSS, NVivo, Gephi, Python, R, Stata",
    "Other": "Terms related to the research methodology but difficult to categorize. Sample: labeling system, indicator system"
}

def build_entity_prompt(entity_definitions):
    return "\n".join(f"{et}: {desc}" for et, desc in entity_definitions.items())

entity_prompt = build_entity_prompt(entity_definitions)

label_mapping = {
    "Question":"Question",
    "Data_Collection": "Data_Collection",
    "Theory": "Theory",
    "Data_Analysis": "Data_Analysis",
    "Dataset": "Dataset",
    "Indicator": "Indicator",
    "Tool": "Tool",
    "Other": "Other"
}

few_shot_examples = '''
Example 1:
Abstract: "Date palm (Phoenix dactylifera) is one of the commonly used polyphenolic rich fruits attributing also to various therapeutic effect in different diseases and disorders. We aimed to study and analyse the global research output related to date palm based on a fact of its large consumption and production in Middle East. We analysed 1,376 papers obtained from SCOPUS database for the period of 2000-11. The study examines major productive countries and their citation impact. We have also analysed inter-collaborative linkages, national priorities of date palm research, besides analysing the characteristics of its high productivity institutions, authors and journal."
Entities: [{"label": "Dataset", "text": "SCOPUS", "start_offset": 358, "end_offset": 363}, {"label": "Indicator", "text": "citation impact", "start_offset": 457, "end_offset": 472}, {"label": "Data_Analysis", "text": "inter-collaborative linkages", "start_offset": 496, "end_offset": 524}]

Example 2:
Abstract: "Purpose Existing studies on crowdsourcing have focused on analyzing isolated contributions by individual participants and thus collaboration dynamics among them are under-investigated. The value of implementing crowdsourcing in problem solving lies in the aggregation of wisdom from a crowd. This study examines how marginality affects collaboration in crowdsourcing. Design/methodology/approach With population level data collected from a global crowdsourcing community (openideo.com), this study applied social network analysis and in particular bipartite exponential random graph modeling (ERGM) to examine how individual level marginality variables (measured as the degree of being located at the margin) affect the team formation in collaboration crowdsourcing. Findings Significant effects of marginality are attributed to collaboration skills, number of projects won, community tenure and geolocation. Marginality effects remain significant after controlling for individual level and team level attributes. However, marginality alone cannot explain collaboration dynamics. Participants with leadership experience or more winning ideas are also more likely to be selected as team members. Originality/value The core contribution this research makes is the conceptualization and definition of marginality as a mechanism in influencing collaborative crowdsourcing. This study conceptualizes marginality as a multidimensional concept and empirically examines its effect on team collaboration, connecting the literature on crowdsourcing to online collaboration."
Entities: [{"label": "Dataset", "text": "global crowdsourcing community (openideo.com)", "start_offset": 438, "end_offset": 485}, {"label": "Data_Analysis", "text": "social network analysis", "start_offset": 506, "end_offset": 529}, {"label": "Data_Analysis", "text": "bipartite exponential random graph modeling (ERGM)", "start_offset": 570, "end_offset": 618}]
'''

def clean_json_output(output):
    if output.startswith("```"):
        lines = output.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        output = "\n".join(lines).strip()
    if not output.endswith("]"):
        pos = output.rfind("}")
        if pos != -1:
            output = output[:pos+1] + "]"
    return output

def extract_entities_gpt(text, timeout=60, max_retries=2):
    retries = 0
    while retries <= max_retries:
        try:
            messages = [
                {"role": "system", "content": "You are an expert in named entity recognition for academic texts."},
                {"role": "assistant", "content": few_shot_examples},
                {"role": "user", "content": (
                    "Extract all entities from the following abstract. "
                    "Return the results as a JSON array, where each element is an object with keys: "
                    "'label', 'text', 'start_offset', 'end_offset'. "
                    "Only extract entities strictly belonging to the following categories:\n\n" +
                    entity_prompt +
                    "\n\nAbstract:\n" + text
                )}
            ]
            headers = {
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            }
            payload = {"model": "gpt-4o", "messages": messages, "max_tokens": 300}
            response = requests.post(API_ENDPOINT, headers=headers, json=payload, timeout=timeout)
            response.raise_for_status()
            response_data = response.json()
            output = response_data["choices"][0]["message"]["content"].strip()
            if not output:
                print("Empty response received from GPT.")
            output = clean_json_output(output)
            try:
                entities = json.loads(output)
                if isinstance(entities, list):
                    return entities
                else:
                    print("GPT output is not a list, got type:", type(entities))
                    return []
            except Exception as e:
                print("Error parsing GPT output as JSON:", e)
                print("Raw GPT output:", output)
                return []
        except requests.exceptions.Timeout:
            print("Request timed out, retrying...")
            retries += 1
            sleep(1)
        except Exception as e:
            print("Error calling GPT API:", e)
            return []
    return []

# ----------------------------
# 2. 读取原始数据 & 只更新entities字段
# ----------------------------
input_file = "/workspace/3_Innovation_method_202506/input_data/2_model_predicted3.jsonl"
output_file = "/workspace/3_Innovation_method_202506/input_data/2_model_predicted_optimized3.jsonl"

BATCH_SIZE = 1000

with open(input_file, 'r', encoding='utf-8') as fin:
    all_samples = [json.loads(line) for line in fin]

buffer = []
processed_count = 0

def write_buffer_to_file(buffer, output_file, mode='a'):
    with open(output_file, mode, encoding='utf-8') as fout:
        for sample in buffer:
            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")

# 先清空输出文件
open(output_file, 'w').close()

for idx, sample in enumerate(all_samples):
    text = sample.get("text") or sample.get("Abstract") or ""
    print(f"Processing PaperID: {sample.get('Paperid','')} ...")
    gpt_entities = extract_entities_gpt(text)
    # 实体内容合法性校验
    filtered_entities = []
    for ent in gpt_entities:
        if not (ent.get("label") and ent.get("text") and isinstance(ent.get("start_offset"), int) and isinstance(ent.get("end_offset"), int)):
            continue
        label = ent["label"]
        label = label_mapping.get(label, label)
        filtered_entities.append({
            "label": label,
            "text": ent["text"],
            "start_offset": ent["start_offset"],
            "end_offset": ent["end_offset"]
        })
    sample["entities"] = filtered_entities
    buffer.append(sample)
    processed_count += 1

    # 每BATCH_SIZE条写一次
    if processed_count % BATCH_SIZE == 0:
        write_buffer_to_file(buffer, output_file, mode='a')
        print(f"Written {processed_count} samples to {output_file}.")
        buffer = []  # 清空缓冲区

    sleep(3)  # 如有需要可调整

# 处理剩下不满1000条的部分
if buffer:
    write_buffer_to_file(buffer, output_file, mode='a')
    print(f"Written final {len(buffer)} samples to {output_file}.")

print(f"All processing complete. Total processed: {processed_count}")