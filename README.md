# zl-back-py - 专利分类预测 Python 服务

基于 BERT 的专利分类模型推理服务，提供医疗器械专利分类预测 API。

## 技术栈

- Python 3.7+
- Flask 2.0+
- PyTorch 1.9+
- Transformers 4.10+
- pandas 1.3+
- scikit-learn 0.24+

## 项目结构

```
zl-back-py/
├── app.py                # Flask 入口，模型缓存
├── back_predict_api.py   # 预测函数，标签映射
├── models/               # 模型架构
│   ├── bert_linear.py
│   ├── bert_textcnn.py
│   └── bert_lstm_attention.py
├── myModel/              # 训练好的模型
│   ├── best_model.pth
│   └── curr_config.yaml
├── preTModel/bert/       # 预训练 BERT
│   └ chinese-roberta-wwm-ext/
├── utils/                # 辅助工具
└── requirements.txt
```

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 启动服务

```bash
python app.py
```

服务启动于 http://localhost:5000

## API 接口

### 健康检查

```http
GET /health
```

响应：
```json
{
  "status": "ok",
  "message": "服务运行正常"
}
```

### 专利分类预测

```http
POST /predict
Content-Type: application/json

{
  "summary": "专利摘要文本"
}
```

响应：
```json
{
  "code": 200,
  "message": "预测成功",
  "data": {
    "pred_label": "医用成像器械",
    "pred_index": 4,
    "pred_probability": 0.95,
    "categories": [
      {
        "name": "医用成像器械",
        "probability": 0.95,
        "index": 4
      }
    ],
    "summary": "输入的摘要文本"
  }
}
```

## 分类类别

共 22 个医疗器械分类类别：

| 序号 | 类别名称 |
|------|----------|
| 0 | 中医器械 |
| 1 | 临床检验器械 |
| 2 | 医用康复器械 |
| 3 | 医用成像器械 |
| 4 | 医用诊察和监护器械 |
| 5 | 医用软件 |
| 6 | 医疗器械消毒灭菌器械 |
| 7 | 口腔科器械 |
| 8 | 呼吸、麻醉和急救器械 |
| 9 | 妇产科、辅助生殖和避孕器械 |
| 10 | 患者承载器械 |
| 11 | 放射治疗器械 |
| 12 | 无源手术器械 |
| 13 | 无源植入器械 |
| 14 | 有源手术器械 |
| 15 | 有源植入器械 |
| 16 | 注输、护理和防护器械 |
| 17 | 物理治疗器械 |
| 18 | 眼科器械 |
| 19 | 神经和心血管手术器械 |
| 20 | 输血、透析和体外循环器械 |
| 21 | 骨科手术器械 |

## 模型说明

- 模型类型: BERT-TextCNN 融合模型
- 预训练模型: chinese-roberta-wwm-ext
- 整体准确率: 81.53%

### 整体性能指标

| 指标 | 数值 |
|------|------|
| Accuracy | 81.53% |
| Macro Avg F1 | 75.27% |
| Weighted Avg F1 | 81.39% |

## 注意事项

1. 首次启动时加载模型较慢
2. 模型缓存在内存中，后续请求更快
3. 确保 `myModel/` 下有模型文件
4. 确保 `preTModel/bert/` 下有预训练模型

## 相关项目

- [zl-front](../zl-front) - Vue 前端
- [zl-back-java](../zl-back-java) - Java 后端