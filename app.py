"""
LangChainVision 应用程序入口点
使用LangGraph和MCP架构的图像分析系统
"""

from src.main import app

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 