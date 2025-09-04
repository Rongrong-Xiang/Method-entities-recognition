import json
import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import AutoTokenizer, AutoConfig, BertModel
from torchcrf import CRF
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# =============================
# 1. 配置与标签
# =============================
entity_types = ["Question", "Theory", "Data_Collection", "Data_Analysis", "Dataset", "Indicator", "Tool", "Other"]
labels_list = ["O"] + [f"{p}-{et}" for et in entity_types for p in ["B", "I"]]
label2id = {l: i for i, l in enumerate(labels_list)}
id2label = {v: k for k, v in label2id.items()}
model_dir = "/workspace/models/scibert_model"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
config = AutoConfig.from_pretrained(model_dir, num_labels=len(label2id), id2label=id2label, label2id=label2id)
hidden_dim = 256
batch_size = 16
max_length = 512
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================
# 2. 数据集与加载
# =============================

class NERDataset(Dataset):
    def __init__(self, samples, tokenizer, label2id, max_length=512):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label2id = label2id
        self.data = [self.convert_sample(sample) for sample in self.samples]

    def convert_sample(self, sample):
        text = sample["text"]
        char_labels = ["O"] * len(text)
        for ent in sample.get("entities", []):
            start, end, label = ent["start_offset"], ent["end_offset"], ent["label"]
            if start < len(text) and end <= len(text):
                char_labels[start] = "B-" + label
                for i in range(start+1, end):
                    char_labels[i] = "I-" + label
        encoding = self.tokenizer(text, return_offsets_mapping=True, truncation=True, max_length=self.max_length)
        offsets = encoding.pop("offset_mapping")
        token_labels = [
            char_labels[offset[0]] if offset != (0, 0) and offset[0] < len(char_labels) else "O"
            for offset in offsets
        ]
        encoding["labels"] = [self.label2id.get(l, self.label2id["O"]) for l in token_labels]
        return encoding

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(sample["input_ids"]) for sample in batch], batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(sample["attention_mask"]) for sample in batch], batch_first=True, padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(sample["labels"]) for sample in batch], batch_first=True, padding_value=label2id["O"])
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

def load_jsonl_data(jsonl_path):
    with open(jsonl_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

# =============================
# 3. 模型定义
# =============================
class SciBERT_BiLSTM_CRF(nn.Module):
    def __init__(self, model_dir, config, hidden_dim, num_labels, dropout_prob=0.1):
        super().__init__()
        self.scibert = BertModel.from_pretrained(model_dir, config=config)
        self.dropout = nn.Dropout(dropout_prob)
        self.lstm = nn.LSTM(config.hidden_size, hidden_dim // 2, num_layers=1,
                            bidirectional=True, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.scibert(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = self.dropout(outputs.last_hidden_state)
        lstm_out, _ = self.lstm(embeddings)
        emissions = self.classifier(lstm_out)
        if labels is not None:
            loss = -self.crf(emissions, labels, mask=attention_mask.bool(), reduction='mean')
            preds = self.crf.decode(emissions, mask=attention_mask.bool())
            return loss, preds
        else:
            preds = self.crf.decode(emissions, mask=attention_mask.bool())
            return preds

# =============================
# 4. 十折交叉验证 + 8:1:1 划分
# =============================

def split_8_1_1(samples, seed=42):
    train_val, test = train_test_split(samples, test_size=0.1, random_state=seed)
    train, val = train_test_split(train_val, test_size=0.1111, random_state=seed)  # 0.1111*0.9 ≈ 0.1
    return train, val, test

def extract_entities(labels_seq):
    entities, start, end, current_label = [], None, None, None
    for i, label in enumerate(labels_seq):
        if label.startswith("B-"):
            if current_label is not None:
                entities.append((start, end, current_label))
            current_label = label[2:]
            start, end = i, i+1
        elif label.startswith("I-") and current_label and label[2:] == current_label:
            end = i+1
        else:
            if current_label is not None:
                entities.append((start, end, current_label))
                current_label = None
    if current_label is not None:
        entities.append((start, end, current_label))
    return entities

def fuzzy_match(ent1, ent2, threshold=0.5):
    if ent1[2] != ent2[2]: return False
    intersection = max(0, min(ent1[1], ent2[1]) - max(ent1[0], ent2[0]))
    union = max(ent1[1], ent2[1]) - min(ent1[0], ent2[0])
    return (intersection / union) >= threshold

def eval_ner(model, dataloader, id2label, device):
    model.eval()
    TP_total = FP_total = FN_total = exact_match = num_samples = 0
    all_true, all_pred = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            preds = model(input_ids, attention_mask)
            labels = labels.cpu().numpy()
            attn = batch["attention_mask"].cpu().numpy()
            for i, pred in enumerate(preds):
                valid_len = int(attn[i].sum())
                true_ids = labels[i][:valid_len]
                pred_ids = np.array(pred)
                true_tokens = [id2label[_id] for _id in true_ids.tolist()]
                pred_tokens = [id2label[_id] for _id in pred_ids.tolist()]
                all_true.extend(true_tokens)
                all_pred.extend(pred_tokens)
                true_ents = extract_entities(true_tokens)
                pred_ents = extract_entities(pred_tokens)
                matched = set()
                TP = 0
                for pe in pred_ents:
                    for idx, te in enumerate(true_ents):
                        if idx not in matched and fuzzy_match(pe, te):
                            TP += 1
                            matched.add(idx)
                            break
                FP = len(pred_ents) - TP
                FN = len(true_ents) - TP
                TP_total += TP
                FP_total += FP
                FN_total += FN
                if TP == len(true_ents) and TP == len(pred_ents):
                    exact_match += 1
                num_samples += 1
    token_scores = precision_recall_fscore_support(all_true, all_pred, average='micro', zero_division=0)
    acc = accuracy_score(all_true, all_pred)
    entity_precision = TP_total/(TP_total+FP_total) if TP_total+FP_total else 0.0
    entity_recall = TP_total/(TP_total+FN_total) if TP_total+FN_total else 0.0
    entity_f1 = 2*entity_precision*entity_recall/(entity_precision+entity_recall) if (entity_precision+entity_recall) else 0.0
    entity_exact = exact_match / num_samples if num_samples else 0.0
    return {
        "token_precision": token_scores[0], "token_recall": token_scores[1], "token_f1": token_scores[2], "token_acc": acc,
        "entity_precision": entity_precision, "entity_recall": entity_recall, "entity_f1": entity_f1, "entity_exact": entity_exact
    }

def train_fold(
    model, train_loader, val_loader, optimizer, device,
    max_epochs=30, early_stop=1, min_epochs=1, early_stop_f1_threshold=0.2
):
    best_f1, best_state, stop_count = 0, None, 0
    early_stop_enabled = False
    for epoch in range(max_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_batch = batch["labels"].to(device)
            loss, _ = model(input_ids, attention_mask, labels=labels_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        val_scores = eval_ner(model, val_loader, id2label, device)
        print(f"Epoch {epoch+1}, loss={total_loss/len(train_loader):.4f}, val_f1={val_scores['entity_f1']:.4f}")
        if val_scores["entity_f1"] > best_f1:
            best_f1 = val_scores["entity_f1"]
            best_state = model.state_dict()
            stop_count = 0
        else:
            stop_count += 1
        # 只有F1超过early_stop_f1_threshold且epoch到达min_epochs后才允许early stopping
        if (epoch+1 >= min_epochs) and (best_f1 >= early_stop_f1_threshold) and (stop_count >= early_stop):
            print("Early stopping.")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_f1

def kfold_cross_val(samples, k=10, batch_size=16, seed=42):
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    fold_results = []
    for fold, (trainval_idx, test_idx) in enumerate(kf.split(samples)):
        trainval = [samples[i] for i in trainval_idx]
        test = [samples[i] for i in test_idx]
        train, val = train_test_split(trainval, test_size=0.1111, random_state=seed)
        trainset = NERDataset(train, tokenizer, label2id, max_length)
        valset = NERDataset(val, tokenizer, label2id, max_length)
        testset = NERDataset(test, tokenizer, label2id, max_length)
        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        model = SciBERT_BiLSTM_CRF(model_dir, config, hidden_dim, len(label2id)).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
        print(f"\n===== Fold {fold+1}/{k} =====")
        train_fold(model, train_loader, val_loader, optimizer, device)
        test_scores = eval_ner(model, test_loader, id2label, device)
        torch.save(model.state_dict(), f"/workspace/models/scibert_bilstm_crf_ner_fold{fold+1}.pth")
        fold_results.append(test_scores)
    avg_scores = {k: np.mean([r[k] for r in fold_results]) for k in fold_results[0]}
    print("\n=== 10-Fold CV Test Results (avg) ===")
    for k, v in avg_scores.items():
        print(f"{k}: {v:.4f}")

# =============================
# 5. 大规模无标签数据预测
# =============================

from collections import OrderedDict

def predict_unlabeled(model_path, data_path, out_path, batch_size=16, device=device, keep_labels=None):
    print(f"Predicting for {data_path} ...")
    model = SciBERT_BiLSTM_CRF(model_dir, config, hidden_dim, len(label2id)).to(device)
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
                # 提取实体
                entities = []
                for ent in extract_entities(pred_labels):
                    start, end, label = ent
                    mention = text[start:end]  # 字符级抽取
                    if keep_labels is None or label in keep_labels:
                        entities.append({
                            "label": label,
                            "text": mention,
                            "start_offset": start,
                            "end_offset": end
                        })
                # --------- 字段筛选与顺序调整 -----------
                src = samples[len(results)]
                new_data = OrderedDict()
                if "Paperid" in src:
                    new_data["Paperid"] = src["Paperid"]
                for key in src:
                    if key in ["id", "relations", "Comments", "entities", "Paperid"]:
                        continue
                    new_data[key] = src[key]
                new_data["entities"] = entities
                results.append(new_data)
    # 保存为 JSONL
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Prediction saved to {out_path}.")

# =============================
# 6. 主流程入口
# =============================

if __name__ == "__main__":
    all_samples = load_jsonl_data("/workspace/3_Innovation_method_202506/input_data/2_manual_labeled_standard.jsonl")
    kfold_cross_val(all_samples, k=2, batch_size=batch_size)

    # 训练全集模型用于最终预测
    print("\nTraining final model on all data for production prediction...")
    trainset = NERDataset(all_samples, tokenizer, label2id, max_length)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    model = SciBERT_BiLSTM_CRF(model_dir, config, hidden_dim, len(label2id)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    train_fold(model, train_loader, train_loader, optimizer, device, max_epochs=20, early_stop=3)  # train/val都用全集即可
    torch.save(model.state_dict(), "/workspace/models/scibert_bilstm_crf_ner.pth")

    # 用全集模型做无标签数据预测
    keep_labels = ["Question", "Theory", "Data_Collection", "Data_Analysis", "Dataset", "Indicator", "Tool", "Other"]
    predict_unlabeled(
        "/workspace/models/scibert_bilstm_crf_ner.pth",
        "/workspace/3_Innovation_method_202506/input_data/1_non_manual_data1.jsonl",
        "/workspace/3_Innovation_method_202506/input_data/2_model_predicted.jsonl",
        keep_labels=keep_labels
    )
    
