import pandas as pd

# 读取原始数据
df = pd.read_csv("agent_results.csv")

# 人工判断后，按顺序给每条打标签
df["label"] = [
    "格式问题",      # 1+1等于几？
    "格式问题",      # 中国的首都是哪里？
    "正确但啰嗦",    # 太阳从哪个方向升起？
    "正确但啰嗦",    # 水的沸点是多少度？
    "答非所问",      # 地球有几个卫星？
    "正确但啰嗦",    # Python 是谁创造的？
    "格式问题",      # 一年有多少个月？
    "格式问题",      # 一周有几天？
    "正确但啰嗦",    # 彩虹有几种颜色？
    "正确但啰嗦",    # 冰是水的什么状态？
]

# 存成新文件（保留原始文件不动）
df.to_csv("agent_results_labeled.csv", index=False, encoding="utf-8-sig")

# 打印出来检查
print("打标签完成，结果：")
print(df[["question", "label"]])