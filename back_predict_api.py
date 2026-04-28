"""
专利预测API模块
从back-predict.py提取可复用的预测函数
"""
import os
import torch
import torch.nn.functional as F
from transformers import BertTokenizer
from omegaconf import OmegaConf
from utils.model_utils import get_model

# === index → 标签名称映射 ===
LABELS = [
    '中医器械', '临床检验器械', '医用康复器械', '医用成像器械', '医用诊察和监护器械', '医用软件',
    '医疗器械消毒灭菌器械', '口腔科器械', '呼吸、麻醉和急救器械', '妇产科、辅助生殖和避孕器械',
    '患者承载器械', '放射治疗器械', '无源手术器械', '无源植入器械', '有源手术器械', '有源植入器械',
    '注输、护理和防护器械', '物理治疗器械', '眼科器械', '神经和心血管手术器械', '输血、透析和体外循环器械', '骨科手术器械'
]

def load_model_and_config(config_path: str, model_path: str, num_labels: int = 22):
    """加载模型和配置"""
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

def load_model_once(model_dir: str = "./myModel"):
    """加载模型（只加载一次，用于API服务）"""
    model_path = os.path.join(model_dir, "best_model.pth")
    config_path = os.path.join(model_dir, "curr_config.yaml")
    model, config, device = load_model_and_config(config_path, model_path, num_labels=len(LABELS))

    # 初始化分词器
    bert_path = os.path.join(config.input.pre_train_model.dir, config.input.pre_train_model.name)
    tokenizer = BertTokenizer.from_pretrained(bert_path)
    print(f"[INFO] 使用预训练分词器: {bert_path}")

    return model, tokenizer, config, device

@torch.no_grad()
def predict_single_text(model, tokenizer, text: str, device, config, max_length=None):
    """
    预测单个文本的分类概率

    Args:
        model: 加载的模型
        tokenizer: 分词器
        text: 要预测的文本
        device: 设备
        config: 配置对象
        max_length: 最大长度，如果为None则使用config中的值

    Returns:
        dict: 包含预测结果和所有类别的概率
    """
    if max_length is None:
        max_length = config.model.BERT.max_length

    # 编码文本
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

    # 预测
    outputs = model(**encoded)
    logits = outputs if isinstance(outputs, torch.Tensor) else outputs[0]
    probs = F.softmax(logits, dim=-1)
    pred_idx = torch.argmax(probs, dim=-1).item()
    pred_label = LABELS[pred_idx]

    # 获取所有类别的概率
    probabilities = probs.squeeze().cpu().tolist()

    # 构建结果，包含所有类别的概率
    categories = []
    for idx, label in enumerate(LABELS):
        categories.append({
            "name": label,
            "probability": probabilities[idx],
            "index": idx
        })

    # 按概率降序排序
    categories.sort(key=lambda x: x["probability"], reverse=True)

    return {
        "pred_label": pred_label,
        "pred_index": pred_idx,
        "pred_probability": probabilities[pred_idx],
        "categories": categories,
        "summary": text
    }

