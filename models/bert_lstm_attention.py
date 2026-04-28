# bert_attn_lstm.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer


class BertAttnLSTMClassifier(nn.Module):
    """
    BERT + Attention + LSTM 分类器

    结构说明：
      - 使用预训练 BERT 提取 token-level embeddings（[B, L, H]）和 CLS 向量（[B, H]）
      - 将 BERT 的 token embedding 过 dropout 后输入 LSTM（可选双向、多层）
      - 对 LSTM 的每个时刻输出使用加性注意力（Bahdanau-style / additive attention），注意力可使用 CLS/mean 特征作为 context query
      - 将注意力上下文向量与 global feature (CLS 或 mean) 融合并经过 MLP 输出 logits

    参数:
      model_path: 预训练 BERT 模型路径或名称
      class_num: 分类数
      max_length: 最大序列长度（用于计算一些可选尺寸，实际计算不严格依赖）
      lstm_hidden_size: LSTM 隐状态维度
      lstm_layers: LSTM 层数
      bidirectional: 是否使用双向 LSTM
      is_fine_tune: 是否微调 BERT
      custom_mask: 是否对 [CLS]/[SEP] 做特殊 mask（与您现有代码兼容）
      dropout_rate: dropout 比例
      freeze_layers: 冻结的 encoder 层数（从头部开始冻结 encoder.layer.0 ... encoder.layer.(freeze_layers-1)）
    """

    def __init__(self,
                 model_path,
                 class_num,
                 max_length,
                 lstm_hidden_size=256,
                 lstm_layers=1,
                 bidirectional=True,
                 is_fine_tune=True,
                 custom_mask=True,
                 dropout_rate=0.1,
                 freeze_layers=0):
        super().__init__()
        print(f"模型参数:\n model_path:{model_path}\nclass_num:{class_num}\nmax_length:{max_length}\nlstm_hidden_size:{lstm_hidden_size}\n"
              f"lstm_layers:{lstm_layers}\nbidirectional:{bidirectional}\ncustom_mask:{custom_mask}\nis_fine_tune:{is_fine_tune}\ndropout_rate:{dropout_rate}\nfreeze_layers:{freeze_layers}")

        # 载入预训练 BERT 与 tokenizer
        self.bert = BertModel.from_pretrained(model_path)
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id

        # 微调/冻结 BERT 参数
        if not is_fine_tune:
            for param in self.bert.parameters():
                param.requires_grad = False
        else:
            # 支持 freeze_layers 参数，冻结前 freeze_layers 个 encoder 层
            for name, param in self.bert.named_parameters():
                if any([f"encoder.layer.{i}." in name for i in range(freeze_layers)]):
                    param.requires_grad = False

        self.custom_mask = custom_mask
        self.hidden_size = self.bert.config.hidden_size
        self.max_length = max_length

        # LSTM
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_layers = lstm_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.lstm_hidden_size,
            num_layers=self.lstm_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
            dropout=dropout_rate if lstm_layers > 1 else 0.0
        )

        # attention parameters (additive attention)
        # 将 LSTM 输出映射到注意力空间
        self.attn_proj_h = nn.Linear(self.lstm_hidden_size * self.num_directions, self.lstm_hidden_size * self.num_directions, bias=False)
        # 将 query (global feature) 映射到注意力空间
        self.attn_proj_q = nn.Linear(self.hidden_size, self.lstm_hidden_size * self.num_directions, bias=False)
        # 最终的注意力评分向量
        self.attn_v = nn.Linear(self.lstm_hidden_size * self.num_directions, 1, bias=False)

        # 融合层与规范化
        fused_size = (self.lstm_hidden_size * self.num_directions) + self.hidden_size  # attention context + cls/mean
        self.fused_norm = nn.LayerNorm(fused_size)
        self.dropout = nn.Dropout(dropout_rate)

        # MLP classifier
        self.mlp = nn.Sequential(
            nn.Linear(fused_size, fused_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fused_size // 2, class_num)
        )

    def _compute_mean_feature(self, embeddings, attention_mask, input_ids):
        """
        embeddings: [B, L, H]
        attention_mask: [B, L]
        input_ids: [B, L]
        返回 mean_feature: [B, H]，对 padding 与 [CLS]/[SEP] 做处理（若 custom_mask=True）
        """
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

            mask = mask.unsqueeze(-1).float()  # [B, L, 1]
            embeddings_masked = embeddings * mask
            sum_emb = embeddings_masked.sum(dim=1)  # [B, H]
            valid_cnt = mask.sum(dim=1).clamp(min=1e-8)  # [B, 1]
            mean_feature = sum_emb / valid_cnt
        else:
            mean_feature = torch.mean(embeddings, dim=1)
        return mean_feature

    def _attention(self, lstm_outputs, query):
        """
        加性注意力（Bahdanau-style）
        lstm_outputs: [B, L, D]  (D = lstm_hidden_size * num_directions)
        query: [B, H]           (H = bert hidden size, e.g. cls_feature)
        返回:
          context: [B, D]
          attn_weights: [B, L]
        过程:
          score_t = v^T * tanh(W_h * h_t + W_q * q)
        """
        # 投影
        # proj_h: [B, L, D]
        proj_h = self.attn_proj_h(lstm_outputs)
        # proj_q: [B, 1, D]
        proj_q = self.attn_proj_q(query).unsqueeze(1)

        # sum -> [B, L, D]
        sum_proj = torch.tanh(proj_h + proj_q)
        # energy -> [B, L, 1]
        energy = self.attn_v(sum_proj)
        energy = energy.squeeze(-1)  # [B, L]

        # mask padding positions: assume lstm_outputs may correspond to padded tokens (we'll receive mask from caller)
        # 注意：调用方应在外部依据 attention_mask 对 energy 做 -inf 掩码；如果未提供，则直接 softmax
        attn_weights = F.softmax(energy, dim=1)  # [B, L]
        attn_weights = attn_weights.unsqueeze(-1)  # [B, L, 1]
        # context: sum_t alpha_t * h_t
        context = torch.sum(attn_weights * lstm_outputs, dim=1)  # [B, D]
        attn_weights = attn_weights.squeeze(-1)  # [B, L]
        return context, attn_weights

    def forward(self, input_ids, attention_mask):
        """
        input_ids: [B, L]
        attention_mask: [B, L]
        返回 logits: [B, class_num]
        """
        # BERT 编码
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = bert_out.last_hidden_state  # [B, L, H]
        cls_feature = embeddings[:, 0, :]        # [B, H]

        # 计算 mean feature（可选忽略 CLS/SEP）
        mean_feature = self._compute_mean_feature(embeddings, attention_mask, input_ids)  # [B, H]

        # 选择 query：这里用 cls_feature 作为查询（也可以改为 mean_feature 或两者拼接后线性投影）
        query = cls_feature  # [B, H]

        # 将 token embedding 输入 LSTM（可以应用 dropout）
        # 传入 LSTM 前我们再使用 dropout 减少过拟合
        lstm_in = self.dropout(embeddings)  # [B, L, H]

        # LSTM 执行
        lstm_out, _ = self.lstm(lstm_in)  # lstm_out: [B, L, D]  D = hidden * num_directions

        # 对 attention 掩码处理：将 padding 位置的 energy 设置为 -inf（以避免 softmax 分配概率）
        # 注意：self._attention 内部没有 mask 参数，因此在这里先计算 energy 并手动 mask 更复杂。为简洁，先计算 attention energy inside _attention（softmax），
        # 并随后将 padding 位置的 attn_weights 归零（通过乘以有效 token mask），并重新归一化。这样实现对 padding 的兼容。
        context, attn_weights = self._attention(lstm_out, query)  # context: [B, D], attn_weights: [B, L]

        # 使用 attention_mask 去除 padding 权重并重新归一化
        if attention_mask is not None:
            # attention_mask: [B, L] (0/1)
            am = attention_mask.float()
            # 如果 custom_mask，注意之前可能将 CLS/SEP 也置为0；那里的 mask 已在 mean_feature 计算时处理，但用于 attention 我们通常只按 padding 掩码
            # 将 attn_weights 在 padding 位置置 0
            attn_weights = attn_weights * am
            # 重新归一化
            denom = attn_weights.sum(dim=1, keepdim=True).clamp(min=1e-8)
            attn_weights = attn_weights / denom
            attn_weights = attn_weights.unsqueeze(-1)  # [B, L, 1]
            context = torch.sum(attn_weights * lstm_out, dim=1)  # [B, D]

        # 融合 context 与 global feature（这里用 cls_feature）
        fused = torch.cat([context, cls_feature], dim=1)  # [B, D + H]
        fused = self.fused_norm(fused)
        fused = self.dropout(fused)

        # MLP 分类
        logits = self.mlp(fused)  # [B, class_num]
        return logits
