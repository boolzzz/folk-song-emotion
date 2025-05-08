import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import pygame
import librosa
import sounddevice as sd
import soundfile as sf
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
from pydub import AudioSegment
import threading
import time
import tempfile

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class EmotionFolkSongAnalyzer:
    def __init__(self):
        # 初始化音频相关参数
        self.sample_rate = 16000  # wav2vec2模型需要16kHz采样率
        self.recording = False
        self.audio_data = None
        self.temp_audio_file = None
        
        # 初始化PyGame音频系统
        pygame.mixer.init()
        
        # 加载模型(从本地)
        print("正在加载模型...")
        try:
            self.processor = Wav2Vec2Processor.from_pretrained("./local_model/processor")
            self.model = Wav2Vec2ForSequenceClassification.from_pretrained("./local_model/model")
            print("模型加载成功！")
        except Exception as e:
            print(f"本地模型加载失败: {e}，将尝试从Hugging Face下载...")
            try:
                model_name = "superb/wav2vec2-base-superb-er"
                self.processor = Wav2Vec2Processor.from_pretrained(model_name)
                self.model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)
                print("模型从Hugging Face加载成功！")
                
                # 保存到本地以备后用
                os.makedirs("./local_model/processor", exist_ok=True)
                os.makedirs("./local_model/model", exist_ok=True)
                self.processor.save_pretrained("./local_model/processor")
                self.model.save_pretrained("./local_model/model")
                print("模型已保存到本地！")
            except Exception as e:
                print(f"模型加载失败: {e}")
        
        # 情感标签
        self.emotions = ["anger", "happy", "sad", "neutral", "fear", "disgust"]
        self.emotion_chinese = {
            "anger": "愤怒",
            "happy": "欢快",
            "sad": "悲伤",
            "neutral": "平静",
            "fear": "恐惧",
            "disgust": "厌恶"
        }
        
        # 民歌情感特征映射
        self.song_emotion_map = {
            "茉莉花": {"happy": 0.6, "neutral": 0.4},
            "康定情歌": {"happy": 0.7, "neutral": 0.3},
            "走西口": {"sad": 0.7, "neutral": 0.3},
            "半个月亮爬上来": {"neutral": 0.5, "sad": 0.3, "happy": 0.2},
            "大漠之歌": {"neutral": 0.4, "sad": 0.4, "happy": 0.2},
            "黄土高坡": {"neutral": 0.5, "sad": 0.3, "happy": 0.2}
        }
        
        # 创建GUI界面
        self.create_gui()
    
    def create_gui(self):
        """创建图形用户界面"""
        self.root = tk.Tk()
        self.root.title("民歌情感分析评分系统")
        self.root.geometry("800x600")
        self.root.configure(bg="#f0f0f0")
        
        # 标题
        title_frame = tk.Frame(self.root, bg="#f0f0f0")
        title_frame.pack(pady=10)
        title_label = tk.Label(
            title_frame, 
            text="民歌情感分析评分系统", 
            font=("SimHei", 24, "bold"),
            bg="#f0f0f0"
        )
        title_label.pack()
        
        # 歌曲选择
        song_frame = tk.Frame(self.root, bg="#f0f0f0")
        song_frame.pack(pady=10)
        
        song_label = tk.Label(
            song_frame, 
            text="选择民歌:", 
            font=("SimHei", 12),
            bg="#f0f0f0"
        )
        song_label.pack(side=tk.LEFT, padx=10)
        
        self.song_var = tk.StringVar()
        song_dropdown = ttk.Combobox(
            song_frame, 
            textvariable=self.song_var,
            values=list(self.song_emotion_map.keys()),
            width=15,
            font=("SimHei", 12)
        )
        song_dropdown.current(0)
        song_dropdown.pack(side=tk.LEFT, padx=10)
        
        # 控制按钮
        control_frame = tk.Frame(self.root, bg="#f0f0f0")
        control_frame.pack(pady=20)
        
        self.record_button = tk.Button(
            control_frame,
            text="开始录音",
            command=self.toggle_recording,
            width=15,
            height=2,
            bg="#4CAF50",
            fg="white",
            font=("SimHei", 12, "bold")
        )
        self.record_button.pack(side=tk.LEFT, padx=10)
        
        self.analyze_button = tk.Button(
            control_frame,
            text="导入音频",
            command=self.import_audio,
            width=15,
            height=2,
            bg="#2196F3",
            fg="white",
            font=("SimHei", 12, "bold")
        )
        self.analyze_button.pack(side=tk.LEFT, padx=10)
        
        self.play_button = tk.Button(
            control_frame,
            text="播放音频",
            command=self.play_audio,
            width=15,
            height=2,
            bg="#FF9800",
            fg="white",
            font=("SimHei", 12, "bold"),
            state=tk.DISABLED
        )
        self.play_button.pack(side=tk.LEFT, padx=10)
        
        # 状态标签
        self.status_var = tk.StringVar()
        self.status_var.set("准备就绪")
        status_label = tk.Label(
            self.root,
            textvariable=self.status_var,
            font=("SimHei", 10),
            bg="#f0f0f0"
        )
        status_label.pack(pady=5)
        
        # 创建结果展示区
        self.result_frame = tk.Frame(self.root, bg="#f0f0f0")
        self.result_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # 左侧情感分析结果
        self.emotion_frame = tk.Frame(self.result_frame, bg="#f0f0f0")
        self.emotion_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        emotion_title = tk.Label(
            self.emotion_frame,
            text="情感分析结果",
            font=("SimHei", 14, "bold"),
            bg="#f0f0f0"
        )
        emotion_title.pack(pady=5)
        
        self.emotion_result_text = tk.Text(
            self.emotion_frame,
            height=10,
            width=30,
            font=("SimHei", 12),
            bg="white"
        )
        self.emotion_result_text.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # 右侧雷达图和评分
        self.visual_frame = tk.Frame(self.result_frame, bg="#f0f0f0")
        self.visual_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.canvas_label = tk.Label(
            self.visual_frame,
            text="情感雷达图",
            font=("SimHei", 14, "bold"),
            bg="#f0f0f0"
        )
        self.canvas_label.pack(pady=5)
        
        self.canvas = tk.Canvas(self.visual_frame, bg="white", width=300, height=300)
        self.canvas.pack(pady=5)
        
        self.score_label = tk.Label(
            self.visual_frame,
            text="最终评分: --",
            font=("SimHei", 16, "bold"),
            bg="#f0f0f0"
        )
        self.score_label.pack(pady=10)
        
        # 启动主循环
        self.root.mainloop()
    
    def toggle_recording(self):
        """切换录音状态"""
        if not self.recording:
            self.start_recording()
        else:
            self.stop_recording()
    
    def start_recording(self):
        """开始录音"""
        self.recording = True
        self.record_button.config(text="停止录音", bg="#f44336")
        self.status_var.set("正在录音...")
        
        # 创建临时文件
        self.temp_audio_file = tempfile.mktemp(suffix='.wav')
        
        # 在新线程中开始录音
        threading.Thread(target=self._record_thread).start()
    
    def _record_thread(self):
        """录音线程函数"""
        try:
            # 录制15秒音频
            duration = 15  # 秒
            self.audio_data = sd.rec(
                int(duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=1,
                dtype='float32'
            )
            
            # 显示倒计时
            for i in range(duration, 0, -1):
                if not self.recording:  # 如果提前停止
                    break
                self.status_var.set(f"正在录音... {i}秒")
                time.sleep(1)
            
            # 确保录音完成
            if self.recording:
                sd.stop()
                self.stop_recording()
        
        except Exception as e:
            messagebox.showerror("录音错误", f"录音过程中出现错误: {str(e)}")
            self.recording = False
            self.record_button.config(text="开始录音", bg="#4CAF50")
            self.status_var.set("准备就绪")
    
    def stop_recording(self):
        """停止录音"""
        if not self.recording:
            return
            
        self.recording = False
        sd.stop()
        self.record_button.config(text="开始录音", bg="#4CAF50")
        self.status_var.set("录音已完成，正在处理...")
        
        # 保存录音数据
        if self.audio_data is not None and len(self.audio_data) > 0:
            sf.write(self.temp_audio_file, self.audio_data, self.sample_rate)
            self.play_button.config(state=tk.NORMAL)
            self.analyze_audio(self.temp_audio_file)
        else:
            self.status_var.set("录音为空，请重试")
    
    def import_audio(self):
        """导入音频文件"""
        file_path = filedialog.askopenfilename(
            title="选择音频文件",
            filetypes=[("音频文件", "*.wav *.mp3 *.ogg")]
        )
        
        if file_path:
            self.status_var.set("正在处理音频...")
            self.temp_audio_file = tempfile.mktemp(suffix='.wav')
            
            # 转换成WAV格式
            try:
                audio = AudioSegment.from_file(file_path)
                audio = audio.set_frame_rate(self.sample_rate)
                audio = audio.set_channels(1)
                audio.export(self.temp_audio_file, format="wav")
                self.play_button.config(state=tk.NORMAL)
                self.analyze_audio(self.temp_audio_file)
            except Exception as e:
                messagebox.showerror("文件处理错误", f"处理音频文件时出错: {str(e)}")
                self.status_var.set("准备就绪")
    
    def play_audio(self):
        """播放录制的音频"""
        if self.temp_audio_file and os.path.exists(self.temp_audio_file):
            pygame.mixer.music.load(self.temp_audio_file)
            pygame.mixer.music.play()
            self.status_var.set("正在播放...")
            
            # 监听播放结束
            def check_music_end():
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
                self.status_var.set("播放结束")
            
            threading.Thread(target=check_music_end).start()
        else:
            messagebox.showwarning("播放错误", "没有可播放的音频")
    
    def analyze_audio(self, audio_file):
        """分析音频情感"""
        self.status_var.set("正在分析情感...")
        
        try:
            # 加载音频文件
            audio, _ = librosa.load(audio_file, sr=self.sample_rate)
            
            # 确保音频长度适合模型（如果太长，截取中间部分）
            max_length = 10 * self.sample_rate  # 最多10秒
            if len(audio) > max_length:
                start = len(audio) // 2 - max_length // 2
                audio = audio[start:start+max_length]
            
            # 模型预测
            inputs = self.processor(audio, sampling_rate=self.sample_rate, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # 获取情感得分
            scores = torch.softmax(outputs.logits, dim=1).numpy()[0]
            emotion_scores = {emotion: score for emotion, score in zip(self.emotions, scores)}
            
            # 显示结果
            self.display_results(emotion_scores)
            
            self.status_var.set("分析完成")
        
        except Exception as e:
            messagebox.showerror("分析错误", f"分析音频时出错: {str(e)}")
            self.status_var.set("分析失败")
    
    def display_results(self, emotion_scores):
        """显示分析结果"""
        # 清空结果文本
        self.emotion_result_text.delete(1.0, tk.END)
        
        # 添加情感得分文本
        self.emotion_result_text.insert(tk.END, "情感识别结果:\n\n")
        
        for emotion, score in emotion_scores.items():
            chinese_name = self.emotion_chinese.get(emotion, emotion)
            percentage = score * 100
            self.emotion_result_text.insert(
                tk.END, 
                f"{chinese_name}: {percentage:.1f}%\n"
            )
        
        # 绘制雷达图
        self.draw_radar_chart(emotion_scores)
        
        # 计算演唱评分
        selected_song = self.song_var.get()
        final_score = self.calculate_score(selected_song, emotion_scores)
        
        # 更新评分显示
        self.score_label.config(text=f"最终评分: {final_score:.1f}")
    
    def draw_radar_chart(self, emotion_scores):
        """绘制情感雷达图"""
        # 清除现有图表
        plt.clf()
        
        # 准备数据
        chinese_labels = [self.emotion_chinese.get(e, e) for e in self.emotions]
        values = [emotion_scores[e] * 100 for e in self.emotions]
        
        # 创建雷达图
        angles = np.linspace(0, 2*np.pi, len(chinese_labels), endpoint=False).tolist()
        values += values[:1]  # 闭合雷达图
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
        ax.plot(angles, values, 'o-', linewidth=2)
        ax.fill(angles, values, alpha=0.25)
        ax.set_thetagrids(np.degrees(angles[:-1]), chinese_labels)
        ax.set_ylim(0, 100)
        ax.grid(True)
        
        # 保存图表到临时文件
        radar_file = tempfile.mktemp(suffix='.png')
        plt.savefig(radar_file, dpi=100, bbox_inches='tight')
        plt.close()
        
        # 显示在GUI上
        radar_img = Image.open(radar_file)
        radar_img = radar_img.resize((300, 300), Image.LANCZOS)
        self.radar_photo = ImageTk.PhotoImage(radar_img)
        
        # 更新Canvas
        self.canvas.delete("all")
        self.canvas.create_image(150, 150, image=self.radar_photo)
        
        # 删除临时文件
        try:
            os.remove(radar_file)
        except:
            pass
    
    def calculate_score(self, song_name, emotion_scores):
        """计算基于情感分析的演唱评分"""
        if song_name not in self.song_emotion_map:
            return 70.0  # 默认分数
        
        target_emotions = self.song_emotion_map[song_name]
        
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
        base_score = np.random.normal(75, 5)  # 模拟原有系统给出的基础得分
        base_score = max(60, min(95, base_score))  # 限制在60-95之间
        
        # 组合情感得分和基础得分
        final_score = base_score * 0.7 + emotion_match_score * 100 * 0.3
        
        return final_score


if __name__ == "__main__":
    app = EmotionFolkSongAnalyzer()
