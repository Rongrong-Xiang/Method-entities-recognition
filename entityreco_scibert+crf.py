import json
import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import AutoTokenizer, AutoConfig, BertPreTrainedModel, BertModel
from torchcrf import CRF
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, jaccard_score
from sklearn.model_selection import KFold, train_test_split

# === 1. 定义 NER 数据集 ===
class NERDataset(Dataset):
    def __init__(self, samples, tokenizer, label2id, max_length=256):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label2id = label2id
        self.data = [self.convert_sample(sample) for sample in self.samples]

    def convert_sample(self, sample):
        text = sample["text"]
        char_labels = ["O"] * len(text)
        for ent in sample.get("entities", []):
            start = ent["start_offset"]
            end = ent["end_offset"]
            label = ent["label"]
            if start < len(text) and end <= len(text):
                char_labels[start] = "B-" + label
                for i in range(start+1, end):
                    char_labels[i] = "I-" + label
        encoding = self.tokenizer(text, return_offsets_mapping=True, truncation=True, max_length=self.max_length)
        offsets = encoding.pop("offset_mapping")
        token_labels = []
        for offset in offsets:
            if offset == (0, 0):
                token_labels.append("O")
            else:
                token_start = offset[0]
                if token_start < len(char_labels):
                    token_labels.append(char_labels[token_start])
                else:
                    token_labels.append("O")
        encoding["labels"] = [self.label2id.get(l, self.label2id["O"]) for l in token_labels]
        return encoding

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(sample["input_ids"]) for sample in batch],
        batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(sample["attention_mask"]) for sample in batch],
        batch_first=True, padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(sample["labels"]) for sample in batch],
        batch_first=True, padding_value=label2id["O"])
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

# === 2. 定义 SciBERT+CRF 模型 ===
class SciBERT_CRF(BertPreTrainedModel):
    def __init__(self, config, num_labels, dropout_prob=0.1):
        super(SciBERT_CRF, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)
        self.init_weights()
    
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state)
        emissions = self.classifier(sequence_output)
        if labels is not None:
            loss = -self.crf(emissions, labels, mask=attention_mask.bool(), reduction='mean')
            preds = self.crf.decode(emissions, mask=attention_mask.bool())
            return loss, preds
        else:
            preds = self.crf.decode(emissions, mask=attention_mask.bool())
            return preds

# === 3. 标签与模型准备 ===
entity_types = ["Quesion","Theory", "Data_Collection", "Data_Analysis", "Dataset", "Indicator", "Tool", "Other"]
labels_list = ["O"]
for et in entity_types:
    labels_list.extend([f"B-{et}", f"I-{et}"])
label2id = {label: idx for idx, label in enumerate(labels_list)}
id2label = {idx: label for label, idx in label2id.items()}

model_dir = "/workspace/models/scibert_model"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
config = AutoConfig.from_pretrained(model_dir, num_labels=len(label2id), id2label=id2label, label2id=label2id)
batch_size = 8
max_length = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 4. 评估函数与实体提取 ===
def ids_to_labels(id_list, id2label):
    return [id2label[i] for i in id_list]

def extract_entities(labels_seq):
    entities = []
    start, end, current_label = None, None, None
    for i, label in enumerate(labels_seq):
        if label.startswith("B-"):
            if current_label is not None:
                entities.append((start, end, current_label))
            current_label = label[2:]
            start = i
            end = i + 1
        elif label.startswith("I-") and current_label is not None:
            if label[2:] == current_label:
                end = i + 1
            else:
                entities.append((start, end, current_label))
                current_label = None
                start, end = None, None
        else:
            if current_label is not None:
                entities.append((start, end, current_label))
                current_label = None
                start, end = None, None
    if current_label is not None:
        entities.append((start, end, current_label))
    return entities

def fuzzy_match(entity1, entity2, threshold=0.5):
    if entity1[2] != entity2[2]:
        return False
    start1, end1 = entity1[0], entity1[1]
    start2, end2 = entity2[0], entity2[1]
    intersection = max(0, min(end1, end2) - max(start1, start2))
    union = max(end1, end2) - min(start1, start2)
    return (intersection / union) >= threshold

def eval_ner(model, dataloader, device):
    model.eval()
    TP_total = FP_total = FN_total = sample_exact_matches = num_samples = 0
    all_true_tokens = []
    all_pred_tokens = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_batch = batch["labels"].to(device)
            preds_batch = model(input_ids, attention_mask)
            labels_batch = labels_batch.cpu().numpy()
            batch_attention = batch["attention_mask"].cpu().numpy()
            for i in range(len(preds_batch)):
                valid_len = int(batch_attention[i].sum())
                true_ids = labels_batch[i][:valid_len]
                pred_ids = np.array(preds_batch[i])
                true_tokens = ids_to_labels(true_ids.tolist(), id2label)
                pred_tokens = ids_to_labels(pred_ids.tolist(), id2label)
                all_true_tokens.extend(true_tokens)
                all_pred_tokens.extend(pred_tokens)
                true_entities = extract_entities(true_tokens)
                pred_entities = extract_entities(pred_tokens)
                matched_true_indices = set()
                TP = 0
                for pred_ent in pred_entities:
                    for idx, true_ent in enumerate(true_entities):
                        if idx not in matched_true_indices and fuzzy_match(pred_ent, true_ent, threshold=0.5):
                            TP += 1
                            matched_true_indices.add(idx)
                            break
                FP = len(pred_entities) - TP
                FN = len(true_entities) - TP
                TP_total += TP
                FP_total += FP
                FN_total += FN
                if TP == len(true_entities) and TP == len(pred_entities):
                    sample_exact_matches += 1
                num_samples += 1
    precision_token, recall_token, f1_token, _ = precision_recall_fscore_support(
        all_true_tokens, all_pred_tokens, average='micro', zero_division=0)
    accuracy_token = accuracy_score(all_true_tokens, all_pred_tokens)
    jaccard_token = jaccard_score(all_true_tokens, all_pred_tokens, average='micro', zero_division=0)
    entity_precision = TP_total / (TP_total + FP_total) if (TP_total + FP_total) > 0 else 0.0
    entity_recall = TP_total / (TP_total + FN_total) if (TP_total + FN_total) > 0 else 0.0
    entity_f1 = 2 * entity_precision * entity_recall / (entity_precision + entity_recall) if (entity_precision + entity_recall) > 0 else 0.0
    entity_exact_accuracy = sample_exact_matches / num_samples if num_samples > 0 else 0.0
    global_jaccard = TP_total / (TP_total + FP_total + FN_total + 1e-8)
    return {
        "token_precision": precision_token,
        "token_recall": recall_token,
        "token_f1": f1_token,
        "token_accuracy": accuracy_token,
        "token_jaccard": jaccard_token,
        "entity_precision": entity_precision,
        "entity_recall": entity_recall,
        "entity_f1": entity_f1,
        "entity_exact_accuracy": entity_exact_accuracy,
        "entity_jaccard": global_jaccard,
    }

# === 5. 十折交叉验证主流程 ===
def load_jsonl_data(jsonl_path):
    with open(jsonl_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def train_one_fold(train_idx, val_idx, all_samples):
    train_samples = [all_samples[i] for i in train_idx]
    val_samples = [all_samples[i] for i in val_idx]
    train_dataset = NERDataset(train_samples, tokenizer, label2id, max_length)
    val_dataset = NERDataset(val_samples, tokenizer, label2id, max_length)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    model = SciBERT_CRF.from_pretrained(model_dir, config=config, num_labels=len(label2id), ignore_mismatched_sizes=True)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    best_f1 = 0
    best_state = None
    stop_count = 0
    max_epochs = 50
    early_stop = 5
    for epoch in range(max_epochs):
        model.train()
        total_loss = 0.0
        for batch in train_dataloader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_batch = batch["labels"].to(device)
            loss, _ = model(input_ids, attention_mask, labels=labels_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_dataloader)
        val_scores = eval_ner(model, val_dataloader, device)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Val Entity F1: {val_scores['entity_f1']:.4f}")
        if val_scores["entity_f1"] > best_f1:
            best_f1 = val_scores["entity_f1"]
            best_state = model.state_dict()
            stop_count = 0
        else:
            stop_count += 1
        if stop_count >= early_stop:
            print("Early stopping.\n")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model

def kfold_cross_val(all_samples, k=10):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_results = []
    for fold, (trainval_idx, test_idx) in enumerate(kf.split(all_samples)):
        print(f"\n====== Fold {fold+1} / {k} ======")
        # 8:1划分train/val
        train_idx, val_idx = train_test_split(trainval_idx, test_size=1/9, random_state=fold)
        model = train_one_fold(train_idx, val_idx, all_samples)
        # 用当前这折的测试集评估
        test_samples = [all_samples[i] for i in test_idx]
        test_dataset = NERDataset(test_samples, tokenizer, label2id, max_length)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        scores = eval_ner(model, test_dataloader, device)
        print("--- Fold Test Results ---")
        for k, v in scores.items():
            print(f"{k}: {v:.4f}")
        fold_results.append(scores)
    # 计算平均
    print("\n====== 10-fold CV Average Results ======")
    for k in fold_results[0]:
        print(f"{k}: {np.mean([r[k] for r in fold_results]):.4f}")
        
###################### 6. 预测 ###########################
def predict_unlabeled(model, model_path, data_path, out_path, batch_size=16, device=device):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    samples = load_jsonl_data(data_path)
    dataset = NERDataset(samples, tokenizer, label2id, max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    results = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            preds = model(input_ids, attention_mask)
            for i, pred_ids in enumerate(preds):
                pred_labels = [id2label[_id] for _id in pred_ids]
                text = samples[len(results)]["text"]
                entities = []
                for ent in extract_entities(pred_labels):
                    start, end, label = ent
                    mention = text[start:end]
                    entities.append({"start": start, "end": end, "label": label, "text": mention})
                results.append({"text": text, "entities": entities})
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Prediction saved to {out_path}.")

# === 7. 主入口 ===
if __name__ == '__main__':
    # 只需将全部标注数据的jsonl文件路径传入
    all_samples = load_jsonl_data("/workspace/3_Innovation_method_202503/input_data/2_manual_data_all_category_filter.jsonl")
    kfold_cross_val(all_samples, k=10)
    # 预测
    model = SciBERT_CRF.from_pretrained(model_dir, config=config, num_labels=len(label2id), ignore_mismatched_sizes=True)
    predict_unlabeled(model, "/workspace/models/scibert_crf_ner_fold1.pth", 
                      "/workspace/3_Innovation_method_202503/input_data/1_non_manual_data1.jsonl",
                      "/workspace/3_Innovation_method_202503/input_data/2_non_manual_predicted.jsonl")