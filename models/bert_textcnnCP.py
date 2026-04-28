# bert_textCNN.py的备份
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer


class BertTextCNNClassifier(nn.Module):
    def __init__(self, model_path, class_num, max_length, kernel_sizes=[2, 3, 4, 5],
                 is_fine_tune=False, custom_mask=True, dropout_rate=0.1,freeze_layers=0):
        print(f"模型参数:\n model_path:{model_path}\nclass_num:{class_num}\nkernel_sizes_{kernel_sizes}\ncustom_mask_{custom_mask}\nis_fine_tune:{is_fine_tune}\ndropout_rate:{dropout_rate}\nfreeze_layers:{freeze_layers}")
        super().__init__()
        # 加载预训练 BERT 模型
        self.bert = BertModel.from_pretrained(model_path)
        self.custom_mask = custom_mask
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id

        # 是否微调 BERT 参数
        if not is_fine_tune:
            for param in self.bert.parameters():
                param.requires_grad = False  # 冻结 BERT
        else:
            for name, param in self.bert.named_parameters():
                # BERT encoder 层命名如 encoder.layer.0 ~ encoder.layer.11
                if any([f"encoder.layer.{i}." in name for i in range(freeze_layers)]):
                    param.requires_grad = False

        self.kernel_sizes = kernel_sizes

        # 构建多个不同卷积核大小的卷积层
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=self.bert.config.hidden_size,
                out_channels=max_length // k,
                kernel_size=k,
                bias=(not self.custom_mask)  # 若mask为真，则禁用bias避免泄漏
            )
            for k in kernel_sizes
        ])
        self.local_h = sum([max_length // k for k in kernel_sizes])

        self.dropout = nn.Dropout(dropout_rate)

        # === LayerNorm after global_fc ===
        self.global_fc = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size * 2, self.local_h),
            nn.ReLU(),
        )
        self.global_norm = nn.LayerNorm(self.local_h)  # <<< LayerNorm added

        # === LayerNorm after CNN local features ===
        self.local_norm = nn.LayerNorm(self.local_h)  # <<< LayerNorm added

        # === LayerNorm before final MLP ===
        self.fused_norm = nn.LayerNorm(self.local_h * 2)  # <<< LayerNorm added

        # MLP for classification
        self.mlp = nn.Sequential(
            nn.Linear(self.local_h * 2, class_num * 2),
            nn.ReLU(),
            self.dropout,
            nn.Linear(class_num * 2, class_num)
        )

    def forward(self, input_ids, attention_mask):
        # === BERT encoding ===
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state  # [B, L, H]
        cls_feature = embeddings[:, 0, :]  # [B, H]

        if self.custom_mask:
            mask = attention_mask.clone()
            if self.cls_token_id is not None:
                mask[input_ids == self.cls_token_id] = 0
            else:
                raise ValueError("[CLS] token id 未找到。")
            if self.sep_token_id is not None:
                mask[input_ids == self.sep_token_id] = 0
            else:
                raise ValueError("[SEP] token id 未找到。")

            mask = mask.unsqueeze(-1).float()
            embeddings = embeddings * mask
            sum_embeddings = embeddings.sum(dim=1)
            valid_token_num = mask.sum(dim=1).clamp(min=1e-8)
            mean_feature = sum_embeddings / valid_token_num
        else:
            mean_feature = torch.mean(embeddings, dim=1)

        # === Global feature ===
        global_feature = torch.cat([cls_feature, mean_feature], dim=1)  # [B, 2H]
        global_feature = self.global_fc(global_feature)                 # [B, local_h]
        global_feature = self.global_norm(global_feature)               # <<< LayerNorm applied

        # === Local feature ===
        x = embeddings.transpose(1, 2)  # [B, H, L]
        conv_outs = [F.relu(conv(x)) for conv in self.convs]
        conv_outs = [F.max_pool1d(c, c.size(2)).squeeze(2) for c in conv_outs]
        local_feature = torch.cat(conv_outs, dim=1)                     # [B, local_h]
        local_feature = self.local_norm(local_feature)                   # <<< LayerNorm applied

        # === Fusion ===
        fused = torch.cat([global_feature, local_feature], dim=1)       # [B, 2 * local_h]
        fused = self.fused_norm(fused)                                   # <<< LayerNorm applied
        fused = self.dropout(fused)

        # === Classification ===
        logits = self.mlp(fused)
        return logits