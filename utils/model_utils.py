# model_utils.py
import os

from models.bert_mlp import BertMLPClassifier
from models.bert_textcnn import BertTextCNNClassifier
from models.bert_linear import BertLinearClassifier
from models.bert_lstm_attention import BertAttnLSTMClassifier
import torch
from sklearn.metrics import accuracy_score, f1_score

def get_model(config, class_num):
    """
    依据配置选择模型
    :param config: yaml配置文件
    :param class_num: 分类数
    :return: 模型实例
    """
    pre_train_model_path = os.path.join(config.input.pre_train_model.dir, config.input.pre_train_model.name)
    # 模型工厂
    model_factory = {
        "Linear":lambda :BertLinearClassifier(
            model_path=pre_train_model_path,
            class_num=class_num,
            is_fine_tune=config.model.BERT.fine_tune,
            dropout_rate=config.model.Linear.dropout,
            freeze_layers=config.model.BERT.freeze_layers,
            feature = config.model.Linear.feature
        ),
        "MLP": lambda: BertMLPClassifier(
            model_path=pre_train_model_path,
            class_num=class_num,
            is_fine_tune=config.model.BERT.fine_tune,
            dropout_rate=config.model.MLP.dropout,
            freeze_layers=config.model.BERT.freeze_layers,
            feature = config.model.MLP.feature
        ),
        "TextCNN": lambda: BertTextCNNClassifier(
            model_path= pre_train_model_path,
            class_num=class_num,
            max_length=config.model.BERT.max_length,
            kernel_sizes=config.model.TextCNN.kernel_sizes,
            is_fine_tune=config.model.BERT.fine_tune,
            custom_mask=config.model.TextCNN.custom_mask,
            dropout_rate=config.model.TextCNN.dropout,
            freeze_layers = config.model.BERT.freeze_layers,
            fusion_type = config.model.TextCNN.fusion_type
        ),
        "LSTM": lambda: BertAttnLSTMClassifier(  # <<< 新增模型入口
            model_path=pre_train_model_path,
            class_num=class_num,
            max_length=config.model.BERT.max_length,
            lstm_hidden_size=config.model.LSTM.hidden_size,
            lstm_layers=config.model.LSTM.layers,
            bidirectional=config.model.LSTM.bidirectional,
            is_fine_tune=config.model.BERT.fine_tune,
            custom_mask=getattr(config.model.Attention, "custom_mask", True),
            dropout_rate=config.model.LSTM.dropout,
            freeze_layers=config.model.BERT.freeze_layers
        ),
        # ...其他模型
    }
    if config.model.name not in model_factory:
        raise ValueError(f"暂不支持模型: {config.model.name}")
    return model_factory[config.model.name]()

def evaluate_model(model, dataloader, device):
    """
    评估模型
    :param model: 模型
    :param dataloader: 数据集
    :param device: 设备名称
    :return: 准确率和F1-score
    """
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for input_ids, attention_mask, labels in dataloader:
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            outputs = model(input_ids, attention_mask)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    return acc, macro_f1