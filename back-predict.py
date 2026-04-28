import os
import torch
import torch.nn.functional as F
import pandas as pd
from transformers import BertTokenizer
from omegaconf import OmegaConf
from sklearn.metrics import classification_report
from utils.model_utils import get_model  # 与训练阶段保持一致

# === index → 标签名称映射 ===
LABELS = [
    '中医器械', '临床检验器械', '医用康复器械', '医用成像器械', '医用诊察和监护器械', '医用软件',
    '医疗器械消毒灭菌器械', '口腔科器械', '呼吸、麻醉和急救器械', '妇产科、辅助生殖和避孕器械',
    '患者承载器械', '放射治疗器械', '无源手术器械', '无源植入器械', '有源手术器械', '有源植入器械',
    '注输、护理和防护器械', '物理治疗器械', '眼科器械', '神经和心血管手术器械', '输血、透析和体外循环器械', '骨科手术器械'
]

def load_model_and_config(config_path: str, model_path: str, num_labels: int = 22):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件未找到: {config_path}")
    config = OmegaConf.load(config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] 使用设备: {device}")

    model = get_model(config, num_labels)
    model = torch.nn.DataParallel(model)  # 与训练保持一致

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件未找到: {model_path}")

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print(f"[INFO] 模型加载成功：{model_path}")
    return model, config, device

@torch.no_grad()
def predict_texts(model, tokenizer, texts, device, max_length=512):
    results = []
    for text in texts:
        encoded = tokenizer(
            text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding="max_length"
        )
        if "token_type_ids" in encoded:
            del encoded["token_type_ids"]
        encoded = {k: v.to(device) for k, v in encoded.items()}

        outputs = model(**encoded)
        logits = outputs if isinstance(outputs, torch.Tensor) else outputs[0]
        probs = F.softmax(logits, dim=-1)
        pred_idx = torch.argmax(probs, dim=-1).item()
        pred_label = LABELS[pred_idx]

        results.append({
            "text": text,
            "pred_index": pred_idx,
            "pred_label": pred_label,
            "probabilities": probs.squeeze().cpu().tolist()
        })
    return results

def predict_from_excel(
        excel_path: str,
        text_col: str = "摘要",
        label_col: str = "标签",
        model_dir: str = "./myModel",
):
    # === Step 1. 加载模型与配置 ===
    model_path = os.path.join(model_dir, "test_best_model.pth")
    config_path = os.path.join(model_dir, "curr_config.yaml")
    model, config, device = load_model_and_config(config_path, model_path, num_labels=len(LABELS))

    # === Step 2. 初始化分词器 ===
    bert_path = os.path.join(config.input.pre_train_model.dir, config.input.pre_train_model.name)
    tokenizer = BertTokenizer.from_pretrained(bert_path)
    print(f"[INFO] 使用预训练分词器: {bert_path}")

    # === Step 3. 读取 Excel ===
    df = pd.read_excel(excel_path)
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"Excel 中未找到指定列: {text_col}, {label_col}")

    texts = df[text_col].astype(str).tolist()
    true_labels = df[label_col].tolist()

    # === Step 4. 执行预测 ===
    results = predict_texts(model, tokenizer, texts, device, max_length=config.model.BERT.max_length)
    pred_labels = [r['pred_label'] for r in results]

    # === Step 5. 输出分类报告 ===
    print("\n=== 分类报告 ===")
    report = classification_report(true_labels, pred_labels, labels=LABELS, zero_division=0)
    print(report)

    # === Step 6. 返回结果 DataFrame 可选 ===
    df['pred_label'] = pred_labels
    df['pred_index'] = [r['pred_index'] for r in results]
    return df

if __name__ == "__main__":
    excel_path = "./myModel/外部验证集.xlsx"  # 你的 Excel 文件路径
    text_col = "摘要"
    label_col = "标签"
    predict_from_excel(excel_path, text_col, label_col)
