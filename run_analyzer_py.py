#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

def check_environment():
    """检查环境并安装缺失的依赖"""
    required_packages = [
        "torch", "torchaudio", "transformers", "librosa", 
        "sounddevice", "numpy", "matplotlib", "pillow", 
        "pygame", "pydub"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"发现缺失的依赖包: {', '.join(missing_packages)}")
        print("正在安装缺失的依赖...")
        
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
        print("依赖安装完成！")

def check_model():
    """检查模型是否已下载到本地"""
    if not os.path.exists("./local_model/model") or not os.path.exists("./local_model/processor"):
        print("本地模型文件不存在，首次运行将从Hugging Face下载模型...")
        
        # 只在模型不存在时才导入这些库，避免导入出错
        try:
            from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
            
            print("下载模型中，请耐心等待...")
            model_name = "superb/wav2vec2-base-superb-er"
            processor = Wav2Vec2Processor.from_pretrained(model_name)
            model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)
            
            # 保存到本地以备后用
            os.makedirs("./local_model/processor", exist_ok=True)
            os.makedirs("./local_model/model", exist_ok=True)
            processor.save_pretrained("./local_model/processor")
            model.save_pretrained("./local_model/model")
            print("模型已下载并保存到本地！")
        except Exception as e:
            print(f"模型下载失败: {e}")
            print("请检查网络连接后重试，或者手动下载模型。")
            input("按Enter键退出...")
            sys.exit(1)
    else:
        print("本地模型文件已存在，无需重新下载。")

def main():
    """主函数"""
    print("="*50)
    print("民歌情感分析评分系统启动器")
    print("="*50)
    
    # 检查环境和模型
    check_environment()
    check_model()
    
    print("\n所有准备工作已完成，正在启动应用...\n")
    
    # 导入并启动主程序
    import emotion_folk_analyzer
    
if __name__ == "__main__":
    main()
