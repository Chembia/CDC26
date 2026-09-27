# import csv
#
# import pandas as pd
# import json
#
# # df = pd.read_csv("complaints.csv")
#
# df = pd.read_csv("complaints.csv", nrows=10000)
#
# for col in df.columns:
#     print(
#         col,
#         "->",
#         df[col].nunique(),
#         "unique values",
#     )


import pandas as pd

input_file = "complaints.csv"
output_file = "complaints_2022_2024_company_product.csv"

# 读取数据
df = pd.read_csv(input_file)

# 转换日期
df["Date received"] = pd.to_datetime(
    df["Date received"],
    errors="coerce"
)

# 筛选 2022-01-01 到 2024-12-31
filtered = df[
    (df["Date received"] >= "2022-01-01") &
    (df["Date received"] < "2025-01-01")
]

# 只保留 Company 和 Product
filtered = filtered[["Company", "Product"]]

# 保存
filtered.to_csv(
    output_file,
    index=False,
    encoding="utf-8-sig"
)

print("完成！")    
print("Rows:", len(filtered))
print("Columns:", filtered.columns.tolist())
print("Saved to:", output_file)