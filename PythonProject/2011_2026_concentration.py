import pandas as pd


# ============================================================
# 1. 读取原始 CFPB 数据
# ============================================================

file_path = "complaints.csv"

df = pd.read_csv(
    file_path,
    low_memory=False
)

print("Original complaints:", len(df))


# ============================================================
# 2. 转换日期
# ============================================================

df["Date received"] = pd.to_datetime(
    df["Date received"],
    errors="coerce"
)


# ============================================================
# 3. 筛选 2011 年开始的数据
# ============================================================

df = df[
    df["Date received"] >= "2011-01-01"
].copy()

print(
    "\nData range:"
)

print(
    df["Date received"].min(),
    "to",
    df["Date received"].max()
)


# ============================================================
# 4. 保留 Credit Card / Prepaid Card 相关 Product
# ============================================================

target_products = [
    "credit card",
    "credit card or prepaid card",
    "prepaid card"
]

df["Product_clean"] = (
    df["Product"]
    .astype(str)
    .str.strip()
    .str.lower()
)

df = df[
    df["Product_clean"].isin(target_products)
].copy()


print(
    "\nCredit / Prepaid Card related complaints:",
    len(df)
)


# ============================================================
# 5. 查看三个 Product 的数量
# ============================================================

print(
    "\nProduct distribution:"
)

print(
    df["Product_clean"]
    .value_counts()
)


# ============================================================
# 6. 创建月份
# ============================================================

df["Month"] = (
    df["Date received"]
    .dt.to_period("M")
)


# ============================================================
# 7. 每个月、每个公司的投诉数量
# ============================================================

company_monthly = (
    df
    .groupby(
        ["Month", "Company"]
    )
    .size()
    .reset_index(
        name="complaints"
    )
)


# ============================================================
# 8. 每个月的投诉总数
# ============================================================

monthly_total = (
    company_monthly
    .groupby("Month")["complaints"]
    .sum()
    .reset_index(
        name="total_complaints"
    )
)


# ============================================================
# 9. 合并每个月的总投诉数
# ============================================================

company_monthly = (
    company_monthly
    .merge(
        monthly_total,
        on="Month",
        how="left"
    )
)


# ============================================================
# 10. 计算每个公司的投诉份额
# ============================================================

company_monthly["share"] = (
    company_monthly["complaints"]
    /
    company_monthly["total_complaints"]
)


# ============================================================
# 11. 计算每个月 HHI
# ============================================================

hhi_monthly = (
    company_monthly
    .groupby("Month")["share"]
    .apply(
        lambda x:
        (x ** 2).sum() * 10000
    )
    .reset_index(
        name="HHI"
    )
)


# ============================================================
# 12. 加入每个月投诉总数
# ============================================================

hhi_monthly = (
    hhi_monthly
    .merge(
        monthly_total,
        on="Month",
        how="left"
    )
)


# ============================================================
# 13. 创建完整的月份序列
# ============================================================

start_month = df["Month"].min()
end_month = df["Month"].max()

all_months = pd.period_range(
    start=start_month,
    end=end_month,
    freq="M"
)


print(
    "\nComplete month range:"
)

print(
    start_month,
    "to",
    end_month
)

print(
    "Total months:",
    len(all_months)
)


# ============================================================
# 14. 保留没有数据的月份
# ============================================================

hhi_monthly = (
    hhi_monthly
    .set_index("Month")
    .reindex(all_months)
    .reset_index()
)


# ============================================================
# 15. 修改 Month 列名称
# ============================================================

hhi_monthly = hhi_monthly.rename(
    columns={
        "index": "Month"
    }
)


# ============================================================
# 16. 排序
# ============================================================

hhi_monthly = (
    hhi_monthly
    .sort_values("Month")
    .reset_index(drop=True)
)


# ============================================================
# 17. 输出结果
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "Monthly Credit / Prepaid Card HHI"
)

print(
    "=" * 70
)

print(
    hhi_monthly.to_string(
        index=False
    )
)


# ============================================================
# 18. 检查没有数据的月份
# ============================================================

missing_months = hhi_monthly[
    hhi_monthly["HHI"].isna()
]


print(
    "\n"
    + "=" * 70
)

print(
    "Months without Credit / Prepaid Card complaints"
)

print(
    "=" * 70
)

print(
    missing_months.to_string(
        index=False
    )
)

print(
    "\nNumber of months without data:",
    len(missing_months)
)


# ============================================================
# 19. 保存 HHI 结果
# ============================================================

output_file = (
    "credit_prepaid_card_monthly_HHI_2011_2026.csv"
)

hhi_monthly.to_csv(
    output_file,
    index=False
)


# ============================================================
# 20. 同时保存公司层面的数据
# ============================================================

company_output_file = (
    "credit_prepaid_card_company_monthly_2011_2026.csv"
)

company_monthly.to_csv(
    company_output_file,
    index=False
)


# ============================================================
# 21. 最终信息
# ============================================================

print(
    "\nResults saved successfully:"
)

print(
    "1.",
    output_file
)

print(
    "2.",
    company_output_file
)

print("\nDone!")