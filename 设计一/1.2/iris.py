# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.datasets import load_iris
from matplotlib.font_manager import FontProperties
import matplotlib

# === 设置全局字体为中文黑体 ===
matplotlib.rcParams['font.family'] = 'SimHei'         # 设置中文字体
matplotlib.rcParams['axes.unicode_minus'] = False     # 显示负号正常

# 1. 初始化设置
plt.close('all')  # 关闭所有现有图形

# 2. 设置字体对象（可选）
try:
    font = FontProperties(fname="C:/Windows/Fonts/simhei.ttf", size=12)
except:
    font = FontProperties(size=12)

# 3. 加载数据
iris = load_iris()
iris_df = pd.DataFrame(data=np.c_[iris.data, iris.target],
                     columns=iris.feature_names + ['Species'])
iris_df['Species'] = iris_df['Species'].map({0: '山鸢尾', 1: '变色鸢尾', 2: '维吉尼亚鸢尾'})

# === 加入更明显的扰动 ===
np.random.seed(42)
iris_df.loc[iris_df['Species'] == '山鸢尾', 'sepal length (cm)'] += np.random.normal(0.2, 0.05, sum(iris_df['Species'] == '山鸢尾'))
iris_df.loc[iris_df['Species'] == '山鸢尾', 'sepal width (cm)'] -= np.random.normal(0.15, 0.05, sum(iris_df['Species'] == '山鸢尾'))

iris_df.loc[iris_df['Species'] == '变色鸢尾', 'petal length (cm)'] += np.random.normal(0.25, 0.1, sum(iris_df['Species'] == '变色鸢尾'))

iris_df.loc[iris_df['Species'] == '维吉尼亚鸢尾', 'petal width (cm)'] += np.random.normal(0.1, 0.1, sum(iris_df['Species'] == '维吉尼亚鸢尾'))

# === 中英文列名映射，用于图形显示 ===
column_map = {
    'sepal length (cm)': '花萼长度(cm)',
    'sepal width (cm)': '花萼宽度(cm)',
    'petal length (cm)': '花瓣长度(cm)',
    'petal width (cm)': '花瓣宽度(cm)'
}

# 4. 图形一：散点图矩阵
pairplot = sns.pairplot(iris_df, hue='Species', palette='husl',
                        vars=list(column_map.keys()))
pairplot.fig.suptitle("1.2.1 徐文杰设计的鸢尾花特征散点图矩阵", y=1.02, fontproperties=font)
for ax in pairplot.axes.flatten():
    if ax is not None:
        xlabel = ax.get_xlabel()
        ylabel = ax.get_ylabel()
        if xlabel in column_map:
            ax.set_xlabel(column_map[xlabel], fontproperties=font)
        if ylabel in column_map:
            ax.set_ylabel(column_map[ylabel], fontproperties=font)
pairplot.fig.tight_layout()

# 5. 图形二：雷达图
plt.figure('1.2.2',figsize=(8, 8))
features = list(column_map.keys())
labels = iris_df['Species'].unique()
angles = np.linspace(0, 2*np.pi, len(features), endpoint=False).tolist()
angles += angles[:1]

ax = plt.subplot(111, polar=True)
palette = sns.color_palette("husl", len(labels))

for idx, label in enumerate(labels):
    values = iris_df[iris_df['Species'] == label][features].mean().tolist()
    values += values[:1]
    ax.plot(angles, values, 'o-', linewidth=2, label=label, color=palette[idx])
    ax.fill(angles, values, color=palette[idx], alpha=0.25)

ax.set_thetagrids(np.degrees(angles[:-1]), [column_map[f] for f in features], fontproperties=font)
ax.set_title("1.2.2 徐文杰设计的鸢尾花四项特征平均值雷达图", fontproperties=font, pad=20)
ax.legend(prop=font, loc='upper right')

# 6. 图形三：箱线图（花萼长度）
plt.figure('1.2.3',figsize=(10, 6))
sns.boxplot(data=iris_df, x='Species', y='sepal length (cm)',
           hue='Species', palette='husl', legend=False)
plt.title("1.2.3 徐文杰设计的花萼长度分布", fontproperties=font)
plt.xlabel("种类", fontproperties=font)
plt.ylabel("花萼长度(cm)", fontproperties=font)

# 7. 显示所有图形
plt.show()
