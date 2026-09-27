import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. 读取公司-月份数据
# ============================================================

file_path = "credit_prepaid_card_company_monthly_2011_2026.csv"

df = pd.read_csv(file_path)

df["Month"] = pd.PeriodIndex(
    df["Month"],
    freq="M"
)

df["Year"] = df["Month"].dt.year


print("=" * 80)
print("Credit / Prepaid Card Concentration Analysis")
print("=" * 80)

print("\nData range:")
print(
    df["Month"].min(),
    "to",
    df["Month"].max()
)

print("\nNumber of observations:")
print(len(df))


# ============================================================
# 2. 检查数据
# ============================================================

print("\nMissing values:")
print(df.isna().sum())


# ============================================================
# 3. 每个月重新计算 market share
# ============================================================

df["share"] = (
    df["complaints"]
    / df["total_complaints"]
)

df["HHI_contribution"] = (
    df["share"] ** 2
) * 10000


# ============================================================
# ============================================================
# ANALYSIS 1
# TOP 5 / TOP 10 COMPANY MARKET SHARE
# ============================================================
# ============================================================


print("\n")
print("=" * 80)
print("ANALYSIS 1: TOP 5 / TOP 10 COMPANY MARKET SHARE")
print("=" * 80)


# ------------------------------------------------------------
# 4. 每个月 Top 5 / Top 10
# ------------------------------------------------------------

monthly_top = []

for month, group in df.groupby("Month"):

    group = group.sort_values(
        "share",
        ascending=False
    )

    top5_share = group.head(5)["share"].sum()
    top10_share = group.head(10)["share"].sum()

    monthly_top.append({
        "Month": month,
        "Top5_share": top5_share,
        "Top10_share": top10_share
    })


monthly_top = pd.DataFrame(monthly_top)


# 转换成百分比

monthly_top["Top5_share_pct"] = (
    monthly_top["Top5_share"] * 100
)

monthly_top["Top10_share_pct"] = (
    monthly_top["Top10_share"] * 100
)


print("\nMonthly Top 5 / Top 10:")
print(
    monthly_top.to_string(index=False)
)


# ------------------------------------------------------------
# 5. Top 5 / Top 10 随时间变化
# ------------------------------------------------------------

plt.figure(figsize=(12, 6))

# 使用整数位置作为 x 轴
x = np.arange(len(monthly_top))

plt.plot(
    x,
    monthly_top["Top5_share_pct"],
    marker="o",
    markersize=3,
    linewidth=1.8,
    label="Top 5"
)

plt.plot(
    x,
    monthly_top["Top10_share_pct"],
    marker="o",
    markersize=3,
    linewidth=1.8,
    label="Top 10"
)


# ------------------------------------------------------------
# 设置年份作为横轴标签
# ------------------------------------------------------------

# 找到每年1月份的位置
year_ticks = []

for year in monthly_top["Month"].dt.year.unique():

    positions = np.where(
        monthly_top["Month"].dt.year == year
    )[0]

    # 找到这一年的第一个月份
    year_ticks.append(positions[0])


# 设置 tick
plt.xticks(
    year_ticks,
    monthly_top.loc[
        year_ticks,
        "Month"
    ].dt.year,
    rotation=0
)


# ------------------------------------------------------------
# Labels
# ------------------------------------------------------------

plt.xlabel("Year")

plt.ylabel(
    "Market Share (%)"
)

plt.title(
    "Top 5 and Top 10 Company Market Share"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "top5_top10_market_share.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# ============================================================
# ANALYSIS 2
# YEARLY CONCENTRATION
# ============================================================
# ============================================================


print("\n")
print("=" * 80)
print("ANALYSIS 2: YEARLY CONCENTRATION")
print("=" * 80)


# ------------------------------------------------------------
# 6. 每个月 HHI
# ------------------------------------------------------------

monthly_hhi = (
    df
    .groupby("Month")["HHI_contribution"]
    .sum()
    .reset_index(
        name="HHI"
    )
)


# ------------------------------------------------------------
# 7. 年度平均 HHI
# ------------------------------------------------------------

yearly_hhi = (
    monthly_hhi
    .assign(
        Year=monthly_hhi["Month"].dt.year
    )
    .groupby("Year")
    .agg(
        Average_HHI=("HHI", "mean"),
        Median_HHI=("HHI", "median"),
        Min_HHI=("HHI", "min"),
        Max_HHI=("HHI", "max"),
        Std_HHI=("HHI", "std")
    )
    .reset_index()
)


print("\nYearly HHI:")
print(
    yearly_hhi.to_string(index=False)
)


# ------------------------------------------------------------
# 8. 年度 Top 5 / Top 10
# ------------------------------------------------------------

yearly_top = []

for year, group in df.groupby("Year"):

    # 每家公司一年投诉数
    company_year = (
        group
        .groupby("Company")["complaints"]
        .sum()
        .reset_index()
    )

    total = company_year["complaints"].sum()

    company_year["share"] = (
        company_year["complaints"]
        / total
    )

    company_year = company_year.sort_values(
        "share",
        ascending=False
    )

    top5 = company_year.head(5)["share"].sum()
    top10 = company_year.head(10)["share"].sum()

    yearly_top.append({
        "Year": year,
        "Top5_share": top5,
        "Top10_share": top10
    })


yearly_top = pd.DataFrame(yearly_top)

yearly_top["Top5_share_pct"] = (
    yearly_top["Top5_share"] * 100
)

yearly_top["Top10_share_pct"] = (
    yearly_top["Top10_share"] * 100
)


print("\nYearly Top 5 / Top 10:")
print(
    yearly_top.to_string(index=False)
)


# ------------------------------------------------------------
# 9. 年度 HHI + Top5 + Top10
# ------------------------------------------------------------

yearly_summary = yearly_hhi.merge(
    yearly_top[
        [
            "Year",
            "Top5_share_pct",
            "Top10_share_pct"
        ]
    ],
    on="Year",
    how="left"
)


print("\n")
print("=" * 80)
print("YEARLY CONCENTRATION SUMMARY")
print("=" * 80)

print(
    yearly_summary.to_string(index=False)
)


# ------------------------------------------------------------
# 10. 绘制年度 HHI
# ------------------------------------------------------------

plt.figure(figsize=(11, 6))

plt.plot(
    yearly_summary["Year"],
    yearly_summary["Average_HHI"],
    marker="o"
)

plt.xlabel("Year")

plt.ylabel("Average HHI")

plt.title(
    "Average Annual HHI: Credit / Prepaid Card Complaints"
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "yearly_HHI.png",
    dpi=300
)

plt.show()


# ------------------------------------------------------------
# 11. 绘制年度 Top 5 / Top 10
# ------------------------------------------------------------

plt.figure(figsize=(11, 6))

plt.plot(
    yearly_summary["Year"],
    yearly_summary["Top5_share_pct"],
    marker="o",
    label="Top 5"
)

plt.plot(
    yearly_summary["Year"],
    yearly_summary["Top10_share_pct"],
    marker="o",
    label="Top 10"
)

plt.xlabel("Year")

plt.ylabel("Market Share (%)")

plt.title(
    "Annual Top 5 and Top 10 Market Share"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "yearly_top5_top10.png",
    dpi=300
)

plt.show()


# ============================================================
# ============================================================
# ANALYSIS 3
# WHICH COMPANIES DRIVE HHI?
# ============================================================
# ============================================================


print("\n")
print("=" * 80)
print("ANALYSIS 3: COMPANIES DRIVING HHI")
print("=" * 80)


# ------------------------------------------------------------
# 12. 每家公司对 HHI 的平均贡献
# ------------------------------------------------------------

company_contribution = (
    df
    .groupby("Company")
    .agg(
        Average_HHI_Contribution=(
            "HHI_contribution",
            "mean"
        ),

        Total_Complaints=(
            "complaints",
            "sum"
        ),

        Average_Market_Share=(
            "share",
            "mean"
        )
    )
    .reset_index()
)


company_contribution[
    "Average_Market_Share_pct"
] = (
    company_contribution[
        "Average_Market_Share"
    ] * 100
)


# ------------------------------------------------------------
# 13. 找出 HHI contribution 最大的公司
# ------------------------------------------------------------

company_contribution = (
    company_contribution
    .sort_values(
        "Average_HHI_Contribution",
        ascending=False
    )
)


print("\nTop companies by average HHI contribution:")

print(
    company_contribution
    .head(20)
    .to_string(index=False)
)


# ------------------------------------------------------------
# 14. Top 20 HHI contribution 图
# ------------------------------------------------------------

top20 = (
    company_contribution
    .head(20)
    .sort_values(
        "Average_HHI_Contribution"
    )
)


plt.figure(figsize=(10, 8))

plt.barh(
    top20["Company"],
    top20["Average_HHI_Contribution"]
)

plt.xlabel(
    "Average HHI Contribution"
)

plt.ylabel(
    "Company"
)

plt.title(
    "Top 20 Companies by Average Contribution to HHI"
)

plt.tight_layout()

plt.savefig(
    "top20_HHI_contributors.png",
    dpi=300
)

plt.show()


# ============================================================
# 15. 分析 HHI 上升期间到底是谁推动
# ============================================================

print("\n")
print("=" * 80)
print("COMPANIES DRIVING HHI INCREASE")
print("=" * 80)


# 每个月公司 HHI contribution

pivot_hhi = (
    df
    .pivot_table(
        index="Month",
        columns="Company",
        values="HHI_contribution",
        aggfunc="sum"
    )
    .fillna(0)
)


# HHI change

monthly_hhi_series = (
    pivot_hhi.sum(axis=1)
)


hhi_change = (
    monthly_hhi_series.diff()
)


# ------------------------------------------------------------
# 16. 计算每家公司对 HHI 月度变化的贡献
# ------------------------------------------------------------

company_changes = pivot_hhi.diff()


# HHI 增加月份

positive_changes = company_changes[
    hhi_change > 0
]


average_positive_contribution = (
    positive_changes
    .mean()
    .sort_values(
        ascending=False
    )
)


print(
    "\nCompanies contributing most during months when HHI increased:"
)

print(
    average_positive_contribution
    .head(20)
)


# ============================================================
# 17. 找出最大的 HHI increase
# ============================================================

largest_increases = (
    hhi_change
    .sort_values(
        ascending=False
    )
    .head(10)
)


print("\nLargest monthly HHI increases:")

print(
    largest_increases
)


# ============================================================
# 18. 对最大的 HHI increase 进行公司分解
# ============================================================

print("\n")
print("=" * 80)
print("DECOMPOSITION OF LARGEST HHI INCREASES")
print("=" * 80)


for month in largest_increases.index:

    previous_month = month - 1

    if previous_month not in pivot_hhi.index:
        continue

    change = (
        pivot_hhi.loc[month]
        -
        pivot_hhi.loc[previous_month]
    )

    change = (
        change
        .sort_values(
            ascending=False
        )
    )

    print("\n")
    print(
        "Period:",
        previous_month,
        "->",
        month
    )

    print(
        "Total HHI increase:",
        hhi_change.loc[month]
    )

    print("\nTop contributors:")

    print(
        change.head(10)
    )


# ============================================================
# 19. 保存结果
# ============================================================

monthly_top.to_csv(
    "monthly_top5_top10.csv",
    index=False
)

yearly_summary.to_csv(
    "yearly_concentration_summary.csv",
    index=False
)

company_contribution.to_csv(
    "company_HHI_contribution.csv",
    index=False
)

average_positive_contribution.to_csv(
    "companies_driving_HHI_increases.csv"
)


# ============================================================
# 20. 完成
# ============================================================

print("\n")
print("=" * 80)

print(
    "ALL ANALYSES COMPLETED"
)

print("=" * 80)

print(
    """
Files generated:

1. monthly_top5_top10.csv
2. yearly_concentration_summary.csv
3. company_HHI_contribution.csv
4. companies_driving_HHI_increases.csv

Figures:

5. top5_top10_market_share.png
6. yearly_HHI.png
7. yearly_top5_top10.png
8. top20_HHI_contributors.png
"""
)

print("Done!")