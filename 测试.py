# 导入pandas
# 读agent_results.csv
# 打印 label
# 列各值出现次数（如果文件里有label列）
# 筛出status == "ok"的行，存成only_ok.csv

import pandas as pd
df=pd.read_csv("agent_results_labeled.csv")
print(df["label"].value_counts())
only_ok=df[df["status"]=="ok"]
only_ok.to_csv("only_ok.csv",index=False,encoding="utf-8-sig")

