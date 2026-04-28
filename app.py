"""
Flask服务，提供专利分类预测API接口
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from back_predict_api import predict_single_text

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 全局变量，用于缓存模型（避免每次请求都加载模型）
_model = None
_tokenizer = None
_config = None
_device = None

def init_model():
    """初始化模型（只加载一次）"""
    global _model, _tokenizer, _config, _device
    if _model is None:
        print("[INFO] 首次加载模型...")
        from back_predict_api import load_model_once
        _model, _tokenizer, _config, _device = load_model_once()
        print("[INFO] 模型加载完成")
    return _model, _tokenizer, _config, _device

@app.route('/health', methods=['GET'])
def health():
    """健康检查接口"""
    return jsonify({"status": "ok", "message": "服务运行正常"})

@app.route('/predict', methods=['POST'])
def predict():
    """专利分类预测接口"""
    try:
        data = request.get_json()
        if not data or 'summary' not in data:
            return jsonify({
                "code": 400,
                "message": "缺少必要参数: summary",
                "data": None
            }), 400
        
        summary = data['summary']
        if not summary or not summary.strip():
            return jsonify({
                "code": 400,
                "message": "专利摘要不能为空",
                "data": None
            }), 400
        
        # 获取模型（如果未加载则加载）
        model, tokenizer, config, device = init_model()
        
        # 执行预测
        result = predict_single_text(model, tokenizer, summary, device, config)
        
        return jsonify({
            "code": 200,
            "message": "预测成功",
            "data": result
        })
        
    except Exception as e:
        print(f"[ERROR] 预测失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "code": 500,
            "message": f"预测失败: {str(e)}",
            "data": None
        }), 500

if __name__ == '__main__':
    # 启动时预加载模型
    print("[INFO] 启动Flask服务，正在加载模型...")
    init_model()
    print("[INFO] Flask服务启动在 http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)

