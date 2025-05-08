#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import librosa
import sounddevice as sd
import soundfile as sf
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 情感标签
EMOTIONS = ["anger", "happy", "sad", "neutral", "fear", "disgust"]
EMOTION_CHINESE = {
    "anger": "愤怒",
    "happy": "欢快",
    "sad": "悲伤",
    "neutral": "平静",
    "fear": "恐惧",
    "disgust": "厌恶"
}

# 民歌情感特征映射
SONG_EMOTION_MAP = {
    "茉莉花": {"happy": 0.6, "neutral": 0.4},
    "康定情歌": {"happy": 0.7, "neutral": 0.3},
    "走西口": {"sad": 0.7, "neutral": 0.3},
    "半个月亮爬上来": {"neutral": 0.5, "sad": 0.3, "happy": 0.2},
    "大漠之歌": {"neutral": 0.4, "sad": 0.4, "happy": 0.2},
    "黄土高坡": {"neutral": 0.5, "sad": 0.3, "happy": 0.2}
}

def load_model():
    """加载预训练模型"""
    print("正在加载情感分析模型...")
    
    try:
        # 尝试从本地加载
        processor = Wav2Vec2Processor.from_pretrained("./local_model/processor")
        model = Wav2Vec2ForSequenceClassification.from_pretrained("./local_model/model")
        print("本地模型加载成功！")
    except Exception as e:
        print(f"本地模型加载失败: {e}")
        print("尝试从Hugging Face下载模型...")
        
        # 在线加载
        model_name = "superb/wav2vec2-base-superb-er"
        processor = Wav2Vec2Processor.from_pretrained(model_name)
        model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)
        print("模型下载成功！")
        
        # 保存到本地
        os.makedirs("./local_model/processor", exist_ok=True)
        os.makedirs("./local_model/model", exist_ok=True)
        processor.save_pretrained("./local_model/processor")
        model.save_pretrained("./local_model/model")
        print("模型已保存到本地！")
    
    return processor, model

def record_audio(duration=10, sample_rate=16000):
    """录制音频"""
    print(f"准备录制 {duration} 秒的音频...")
    print("3秒后开始录音...")
    
    for i in range(3, 0, -1):
        print(f"{i}...")
        time.sleep(1)
    
    print("开始录音，请演唱...")
    
    # 录制音频
    audio_data = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype='float32'
    )
    
    # 显示倒计时
    for i in range(duration, 0, -1):
        print(f"剩余时间: {i}秒")
        time.sleep(1)
    
    sd.stop()
    print("录音完成！")
    
    # 保存到临时文件
    temp_file = "temp_recording.wav"
    sf.write(temp_file, audio_data, sample_rate)
    
    return temp_file, audio_data

def load_audio_file(file_path, sample_rate=16000):
    """加载音频文件"""
    print(f"加载音频文件: {file_path}")
    audio, _ = librosa.load(file_path, sr=sample_rate)
    return audio

def analyze_emotion(audio, processor, model, sample_rate=16000):
    """分析音频情感"""
    print("正在分析情感...")
    
    # 确保音频长度适合模型（如果太长，截取中间部分）
    max_length = 10 * sample_rate  # 最多10秒
    if len(audio) > max_length:
        start = len(audio) // 2 - max_length // 2
        audio = audio[start:start+max_length]
    
    # 模型预测
    inputs = processor(audio, sampling_rate=sample_rate, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 获取情感得分
    scores = torch.softmax(outputs.logits, dim=1).numpy()[0]
    emotion_scores = {emotion: score for emotion, score in zip(EMOTIONS, scores)}
    
    return emotion_scores

def display_results(emotion_scores, song_name=None):
    """显示分析结果"""
    print("\n===== 情感分析结果 =====")
    
    for emotion, score in emotion_scores.items():
        chinese_name = EMOTION_CHINESE.get(emotion, emotion)
        percentage = score * 100
        print(f"{chinese_name}: {percentage:.1f}%")
    
    # 如果指定了歌曲，计算评分
    if song_name and song_name in SONG_EMOTION_MAP:
        final_score = calculate_score(song_name, emotion_scores)
        print(f"\n歌曲 '{song_name}' 的最终评分: {final_score:.1f}")
    
    # 绘制雷达图
    draw_radar_chart(emotion_scores)

def calculate_score(song_name, emotion_scores):
    """计算基于情感分析的演唱评分"""
    if song_name not in SONG_EMOTION_MAP:
        return 70.0  # 默认分数
    
    target_emotions = SONG_EMOTION_MAP[song_name]
    
    # 计算情感匹配度得分
    emotion_match_score = 0
    for emotion, target_weight in target_emotions.items():
        if emotion in emotion_scores:
            # 计算实际情感与目标情感的匹配度
            actual = emotion_scores[emotion]
            match = 1.0 - abs(actual - target_weight)  # 值越接近，得分越高
            emotion_match_score += match * target_weight
    
    # 归一化情感匹配度得分
    total_weight = sum(target_emotions.values())
    if total_weight > 0:
        emotion_match_score = emotion_match_score / total_weight
    
    # 假设原有评分系统提供了音准等基础得分(这里模拟)
    base_score = np.random.normal(78, 5)  # 模拟原有系统给出的基础得分
    base_score = max(60, min(95, base_score))  # 限制在60-95之间
    
    # 组合情感得分和基础得分
    final_score = base_score * 0.7 + emotion_match_score * 100 * 0.3
    
    return final_score

def draw_radar_chart(emotion_scores):
    """绘制情感雷达图"""
    # 准备数据
    chinese_labels = [EMOTION_CHINESE.get(e, e) for e in EMOTIONS]
    values = [emotion_scores[e] * 100 for e in EMOTIONS]
    
    # 创建雷达图
    angles = np.linspace(0, 2*np.pi, len(chinese_labels), endpoint=False).tolist()
    values += values[:1]  # 闭合雷达图
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, values, 'o-', linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    ax.set_thetagrids(np.degrees(angles[:-1]), chinese_labels)
    ax.set_ylim(0, 100)
    ax.grid(True)
    
    # 保存和显示
    plt.savefig("emotion_radar.png")
    print("情感雷达图已保存为 'emotion_radar.png'")
    
    try:
        plt.show()  # 尝试显示图表（如果在有GUI的环境中）
    except:
        print("无法显示图表，但已保存为文件")

def main():
    """主函数"""
    print("="*50)
    print("民歌情感分析评分系统 (简化演示版)")
    print("="*50)
    
    # 加载模型
    processor, model = load_model()
    
    # 显示可用歌曲
    print("\n可选民歌:")
    for i, song in enumerate(SONG_EMOTION_MAP.keys(), 1):
        print(f"{i}. {song}")
    
    # 选择歌曲
    while True:
        try:
            choice = int(input("\n请选择歌曲编号: "))
            if 1 <= choice <= len(SONG_EMOTION_MAP):
                song_name = list(SONG_EMOTION_MAP.keys())[choice-1]
                print(f"您选择了: {song_name}")
                break
            else:
                print("无效的选择，请重试")
        except ValueError:
            print("请输入数字")
    
    # 选择输入方式
    while True:
        print("\n请选择输入方式:")
        print("1. 现场录音")
        print("2. 导入音频文件")
        
        try:
            mode = int(input("输入选择: "))
            if mode in [1, 2]:
                break
            else:
                print("无效的选择，请重试")
        except ValueError:
            print("请输入数字")
    
    # 获取音频
    if mode == 1:
        # 录制音频
        file_path, audio_data = record_audio()
        audio = audio_data.flatten()  # 展平数组
    else:
        # 导入音频文件
        while True:
            file_path = input("输入音频文件路径: ")
            if os.path.exists(file_path):
                audio = load_audio_file(file_path)
                break
            else:
                print("文件不存在，请重试")
    
    # 分析情感
    emotion_scores = analyze_emotion(audio, processor, model)
    
    # 显示结果
    display_results(emotion_scores, song_name)
    
    input("\n按Enter键退出...")

if __name__ == "__main__":
    main()