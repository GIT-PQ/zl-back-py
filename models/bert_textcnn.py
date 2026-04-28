# bert_textCNN.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer

class BertTextCNNClassifier(nn.Module):
    def __init__(self, model_path, class_num, max_length=512, kernel_sizes=[2,3,4,5],
                 is_fine_tune=False, custom_mask=True, dropout_rate=0.1, freeze_layers=0,
                 fusion_type="global+local"):
        print(f"模型参数:\n model_path:{model_path}\nclass_num:{class_num}\nkernel_sizes_{kernel_sizes}\ncustom_mask_{custom_mask}\nis_fine_tune:{is_fine_tune}\ndropout_rate:{dropout_rate}\nfreeze_layers:{freeze_layers}\nfusion_type:{fusion_type}\n")
        super().__init__()
        self.fusion_type = fusion_type
        self.custom_mask = custom_mask
        self.kernel_sizes = kernel_sizes

        # --- BERT 加载与微调 ---
        self.bert = BertModel.from_pretrained(model_path)
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id

        if not is_fine_tune:
            for param in self.bert.parameters():
                param.requires_grad = False
        else:
            for name, param in self.bert.named_parameters():
                if any([f"encoder.layer.{i}." in name for i in range(freeze_layers)]):
                    param.requires_grad = False

        # --- CNN ---
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=self.bert.config.hidden_size,
                out_channels=max_length // k,
                kernel_size=k,
                bias=(not self.custom_mask)
            ) for k in kernel_sizes
        ])
        self.local_h = sum([max_length // k for k in kernel_sizes])

        # --- 全局映射 & LayerNorm ---
        self.global_fc = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size * 2, self.local_h),
            nn.ReLU()
        )
        self.global_norm = nn.LayerNorm(self.local_h)
        self.local_norm = nn.LayerNorm(self.local_h)
        self.fused_norm = nn.LayerNorm(self.local_h * 2)

        self.dropout = nn.Dropout(dropout_rate)

        # --- MLP 输入维度根据 fusion_type 自动调整 ---
        if fusion_type == "global":
            mlp_input_dim = self.local_h
        elif fusion_type == "local":
            mlp_input_dim = self.local_h
        elif fusion_type == "global+local":
            mlp_input_dim = self.local_h * 2
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")

        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, class_num * 2),
            nn.ReLU(),
            self.dropout,
            nn.Linear(class_num * 2, class_num)
        )

    def forward(self, input_ids, attention_mask):
        # === BERT encoding ===
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state
        cls_feature = embeddings[:, 0, :]

        # === Mean pooling ===
        if self.custom_mask:
            mask = attention_mask.clone()
            mask[input_ids == self.cls_token_id] = 0
            mask[input_ids == self.sep_token_id] = 0
            mask = mask.unsqueeze(-1).float()
            embeddings_masked = embeddings * mask
            sum_embeddings = embeddings_masked.sum(dim=1)
            valid_token_num = mask.sum(dim=1).clamp(min=1e-8)
            mean_feature = sum_embeddings / valid_token_num
        else:
            mean_feature = torch.mean(embeddings, dim=1)

        # === Global feature ===
        global_feature = torch.cat([cls_feature, mean_feature], dim=1)
        global_feature = self.global_fc(global_feature)
        global_feature = self.global_norm(global_feature)

        # === Local feature ===
        x = embeddings.transpose(1, 2)
        conv_outs = [F.relu(conv(x)) for conv in self.convs]
        conv_outs = [F.max_pool1d(c, c.size(2)).squeeze(2) for c in conv_outs]
        local_feature = torch.cat(conv_outs, dim=1)
        local_feature = self.local_norm(local_feature)

        # === Fusion ===
        if self.fusion_type == "global":
            fused = global_feature
        elif self.fusion_type == "local":
            fused = local_feature
        elif self.fusion_type == "global+local":
            fused = torch.cat([global_feature, local_feature], dim=1)
            fused = self.fused_norm(fused)
        else:
            raise ValueError(f"Unknown fusion_type: {self.fusion_type}")

        fused = self.dropout(fused)
        logits = self.mlp(fused)
        return logits
