import pandas as pd

df = pd.read_csv("agent_results_labeled.csv")

# 按规范优先级判断
def auto_label(row):
    answer = row["answer"]
    # 1. 先判"答非所问"——用关键词信号
    if "取决于" in answer or "看情况" in answer:
        return "答非所问"
    # 2. 再判"格式问题"
    if "**" in answer:
        return "格式问题"
    # 3. 再判"正确但啰嗦"
    if len(answer) > 100:
        return "正确但啰嗦"
    # 4. 剩下判"正确"
    return "正确"

df["auto_label"] = df.apply(auto_label, axis=1)

# 对比
print(df[["question", "label", "auto_label"]])

# 一致率
match = (df["label"] == df["auto_label"]).sum()
print(f"\n一致率: {match}/{len(df)} = {match/len(df):.0%}")

# 不一致的
print("\n不一致的条目：")
print(df[df["label"] != df["auto_label"]][["question", "label", "auto_label"]])