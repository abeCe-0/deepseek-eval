from openai import OpenAI
from dotenv import load_dotenv
import pandas as pd
import time
import os

load_dotenv(r"D:\python\.env")


# 1. 配置客户端
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1"
)

# 2. 准备一批测试问题（这就是的“评测集
questions = [
    "1+1等于几？",
    "中国的首都是哪里？",
    "太阳从哪个方向升起？",
    "水的沸点是多少度？",
    "地球有几个卫星？",
    "Python 是谁创造的？",
    "一年有多少个月？",
    "一周有几天？",
    "彩虹有几种颜色？",
    "冰是水的什么状态？",
]

# 3. 逐条调用模型，收集回答
results = []
for i, q in enumerate(questions):
    print(f"[{i+1}/{len(questions)}] 正在问：{q}")
    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": q}]
        )
        answer = resp.choices[0].message.content
        results.append({"question": q, "answer": answer, "status": "ok"})
    except Exception as e:
        results.append({"question": q, "answer": str(e), "status": "error"})
    time.sleep(0.5)   # 避免请求太快被限流

# 4. 存成 CSV
df = pd.DataFrame(results)
df.to_csv("agent_results.csv", index=False, encoding="utf-8-sig")
print("\n完成，结果已存入 agent_results.csv")
print(df)