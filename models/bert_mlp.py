# bert_mlp.py
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

class BertMLPClassifier(nn.Module):
    def __init__(self, model_path, class_num, is_fine_tune,dropout_rate=0.1,freeze_layers=0,feature='cls'):
        print(f"模型参数:\n model_path:{model_path}\nclass_num:{class_num}\nis_fine_tune:{is_fine_tune}\ndropout_rate:{dropout_rate}\nfreeze_layers:{freeze_layers}")
        super().__init__()
        # 加载预训练的 BERT 模型
        self.bert = BertModel.from_pretrained(model_path)
        self.feature = feature
        # 是否对 BERT 参数进行微调
        if not is_fine_tune:
            for param in self.bert.parameters():
                param.requires_grad = False  # 冻结 BERT 参数
        else:
            for name, param in self.bert.named_parameters():
                # BERT encoder 层命名如 encoder.layer.0 ~ encoder.layer.11
                if any([f"encoder.layer.{i}." in name for i in range(freeze_layers)]):
                    param.requires_grad = False

        self.dropout = nn.Dropout(dropout_rate)  # Dropout 防止过拟合

        # 定义 MLP（包含两个线性层）
        self.mlp = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size,class_num*2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(class_num*2, class_num)
        )

    def forward(self, input_ids, attention_mask):
        # 获取 BERT 的输出
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        if self.feature == 'cls':
            embeddings = outputs.last_hidden_state  # [B, L, H]
            feature = embeddings[:, 0, :]  # [B, H]
        else:
            feature = outputs.pooler_output

        x = self.dropout(feature)

        return self.mlp(x)  # 输出类别 logits