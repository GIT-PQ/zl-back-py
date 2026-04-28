# 专利分类预测 Python 服务

## 环境要求

- Python 3.7+
- PyTorch
- transformers
- Flask

## 安装依赖

```bash
pip install -r requirements.txt
```

## 启动服务

```bash
python app.py
```

服务将在 `http://localhost:5000` 启动

## API 接口

### 健康检查
- **URL**: `/health`
- **方法**: `GET`
- **响应**: 
```json
{
  "status": "ok",
  "message": "服务运行正常"
}
```

### 专利分类预测
- **URL**: `/predict`
- **方法**: `POST`
- **请求体**:
```json
{
  "summary": "专利摘要文本"
}
```

- **响应**:
```json
{
  "code": 200,
  "message": "预测成功",
  "data": {
    "pred_label": "预测的类别名称",
    "pred_index": 0,
    "pred_probability": 0.95,
    "categories": [
      {
        "name": "类别名称",
        "probability": 0.95,
        "index": 0
      }
    ],
    "summary": "输入的摘要文本"
  }
}
```

## 注意事项

1. 首次启动时会加载模型，可能需要一些时间
2. 模型会缓存在内存中，后续请求会更快
3. 确保 `myModel` 目录下有模型文件和配置文件

