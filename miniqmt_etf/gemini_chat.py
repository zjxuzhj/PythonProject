import requests
import json
import os
import sys

class GeminiChat:
    def __init__(self, api_key, model_name=None):
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        
        # 如果未指定模型，自动选择
        if not self.model_name:
            self.model_name = self._auto_select_model()
            
        print(f"当前使用模型: {self.model_name}")

    def _auto_select_model(self):
        """自动选择最佳可用模型"""
        try:
            url = f"{self.base_url}/models?key={self.api_key}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            models = response.json().get('models', [])
            
            # 优先顺序: 2.5-flash -> 1.5-flash -> flash -> pro
            target_models = ["gemini-3-pro-preview","gemini-3-flash-preview","gemini-2.5-flash", "gemini-1.5-flash", "flash", "gemini-pro"]
            
            for target in target_models:
                found = next((m['name'] for m in models if target in m['name'] and 'generateContent' in m.get('supportedGenerationMethods', [])), None)
                if found:
                    return found
            
            # 兜底
            found = next((m['name'] for m in models if 'generateContent' in m.get('supportedGenerationMethods', [])), None)
            return found or "models/gemini-1.5-flash"
            
        except Exception as e:
            print(f"获取模型列表失败: {e}，将使用默认模型。")
            return "models/gemini-1.5-flash"

    def chat(self, user_input):
        """发送消息并获取回复"""
        url = f"{self.base_url}/{self.model_name}:generateContent?key={self.api_key}"
        
        payload = {
            "contents": [{
                "parts": [{"text": user_input}]
            }]
        }
        
        try:
            # 增加 verify=False 可选（如果用户环境有SSL问题），但默认还是保持默认
            # 如果是 windows 有时候会有编码问题，强制设置 encoding
            response = requests.post(url, headers={"Content-Type": "application/json"}, json=payload, timeout=60)
            response.encoding = 'utf-8' # 确保中文显示正常
            
            if response.status_code != 200:
                return f"Error: {response.status_code} - {response.text}"
                
            result = response.json()
            try:
                text = result['candidates'][0]['content']['parts'][0]['text']
                return text
            except (KeyError, IndexError):
                return f"Error parsing response: {json.dumps(result)}"
                
        except Exception as e:
            return f"Network Error: {str(e)}"

def main():
    # 警告：Key 硬编码仅用于演示，生产环境请使用环境变量
    API_KEY = "AIzaSyBlFw-ndswDk4gbDmzUlfs5bnFwPrGEK2c"
    
    chat_bot = GeminiChat(API_KEY)
    
    # 支持命令行参数直接提问
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        print(f"\nUser: {question}")
        answer = chat_bot.chat(question)
        print(f"Gemini: {answer}\n")
        return

    print("\n=== Gemini AI 问答助手 ===")
    print("输入 'quit', 'exit' 或 'q' 退出程序")
    print("==========================")
    
    while True:
        try:
            user_input = input("\n请输入问题: ").strip()
            if not user_input:
                continue
                
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("再见！")
                break
                
            print("Gemini 思考中...", end="", flush=True)
            answer = chat_bot.chat(user_input)
            # 清除 "Gemini 思考中..." 并换行
            print("\r" + " " * 20 + "\r", end="", flush=True)
            print(f"Gemini: {answer}")
            
        except KeyboardInterrupt:
            print("\n再见！")
            break
        except Exception as e:
            print(f"\n发生错误: {e}")

if __name__ == "__main__":
    main()
