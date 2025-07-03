# -*- coding: utf-8 -*-
"""
Created on 2025.06.30 10.00
@author:徐文杰/智科20222/22331050217
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体（解决中文乱码问题）
rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

data = np.array([-2.67,-3.55,-1.24,-0.98,-0.79,-2.85,-2.76,-3.73,-3.54,-2.27,-3.45,
               -3.08,-1.58,-1.49,-0.74,-0.42,-1.12,4.25,-3.99,2.88,-0.98,0.79,1.19,3.07])

def function_norma(x, u, sd):
    """正态分布概率密度函数"""
    return np.exp(-(x-u)**2/(2*sd**2))/(sd * np.sqrt(2 * np.pi))

# 参数设置
pw1 = 0.9  # 正常状态的先验概率
pw2 = 0.1  # 异常状态的先验概率
pxw1 = function_norma(data, -2, 1.5)  # 正常类条件概率
pxw2 = function_norma(data, 2, 2)     # 异常类条件概率（修正均值）
px = (pxw1*pw1 + pxw2*pw2)           # 全概率
pwx1 = (pxw1*pw1)/px                 # 正常后验概率
pwx2 = 1 - pwx1                      # 异常后验概率

# 分类决策
new_array = np.where(pwx1 > pwx2, 1, 2)  # 向量化操作替代循环

# 打印结果
print("第一类的后验概率:\n", np.round(pwx1, 4))
print("第二类的后验概率:\n", np.round(pwx2, 4))
print("\n基于最小错误率的贝叶斯分类结果:")
print("1 表示正常类，2 表示异常类")
print(new_array)

# 可视化设置
plt.figure("1.1 细胞状态分类结果", figsize=(10, 6))  # 设置窗口标题

# 绘制原始数据折线图
plt.plot(np.arange(1,25), data, 'b-', marker='o', label='观测值')

# 绘制分类结果散点图
plt.scatter(np.arange(1,25)[new_array==1], data[new_array==1],
            s=80, c='lime', marker='^', label='正常细胞')
plt.scatter(np.arange(1,25)[new_array==2], data[new_array==2],
            s=80, c='red', marker='s', label='异常细胞')

# 添加图表元素
plt.title("1.1 徐文杰设计的细胞状态贝叶斯分类结果", fontsize=14, pad=20)
plt.xlabel("样本序号", fontsize=12)
plt.ylabel("观测值", fontsize=12)
plt.xticks(range(1,25))
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=12)

plt.tight_layout()
plt.show()