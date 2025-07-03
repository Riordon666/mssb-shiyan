import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 设置中文字体
font_path = "C:/Windows/Fonts/msyh.ttc"
font_prop = font_manager.FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()

# 原始数据
x1 = [0.23, 1.52, 0.65, 0.77, 1.05, 1.19, 0.29, 0.25, 0.66, 0.56, 0.90, 0.13, -0.54, 0.94, -0.21, 0.05, -0.08, 0.73,
      0.33, 1.06, -0.02, 0.11, 0.31, 0.66]
y1 = [2.34, 2.19, 1.67, 1.63, 1.78, 2.01, 2.06, 2.12, 2.47, 1.51, 1.96, 1.83, 1.87, 2.29, 1.77, 2.39, 1.56, 1.93, 2.20,
      2.45, 1.75, 1.69, 2.48, 1.72]
x2 = [1.40, 1.23, 2.08, 1.16, 1.37, 1.18, 1.76, 1.97, 2.41, 2.58, 2.84, 1.95, 1.25, 1.28, 1.26, 2.01, 2.18, 1.79, 1.33,
      1.15, 1.70, 1.59, 2.93, 1.46]
y2 = [1.02, 0.96, 0.91, 1.49, 0.82, 0.93, 1.14, 1.06, 0.81, 1.28, 1.46, 1.43, 0.71, 1.29, 1.37, 0.93, 1.22, 1.18, 0.87,
      0.55, 0.51, 0.99, 0.91, 0.71]

# 增广矩阵并处理第二类样本
w1 = np.vstack([np.ones(24), x1, y1])  # 第一类样本: (1, x, y)
w2 = np.vstack([-np.ones(24), -np.array(x2), -np.array(y2)])  # 第二类样本: (-1, -x, -y)

# 合并样本
all_samples = np.hstack([w1, w2])

# 初始化权向量
a = np.array([1.0, 1.0, 1.0])

# 设置学习率
learning_rate = 0.01
max_iterations = 1000

# 感知器算法
k = 0
converged = False
for k in range(max_iterations):
    misclassified = 0
    for i in range(all_samples.shape[1]):
        if np.dot(a, all_samples[:, i]) <= 0:  # 分类错误
            a += learning_rate * all_samples[:, i]  # 更新权向量
            misclassified += 1
    if misclassified == 0:
        converged = True
        break

print(f"迭代次数: {k}")
print("最终权向量 a:")
print(a)

# 绘图
plt.figure(figsize=(12, 8))
plt.plot(x1, y1, '+r', markersize=8, label="第一类样本点 w1")
plt.plot(x2, y2, 'bs', markersize=8, label="第二类样本点 w2")

# 计算分界面 (a0 + a1*x + a2*y = 0)
xmin = min(min(x1), min(x2)) - 1
xmax = max(max(x1), max(x2)) + 1
x = np.linspace(xmin, xmax, 100)
y = (-a[0] - a[1] * x) / a[2]
plt.plot(x, y, '-k', linewidth=2, label="决策边界")

# 增强的测试数据
test_points = np.array([
    [1.0, 1.2, 1.5],[1.0, 1.0, 1.2],
    [1.0, 2.5, 1.1],[1.0, 1.9, 1.0],
    [1.0, 2.7, 2.8],[1.0, 0.23, 2.33],
    [1.0, 1.5, 0.5],[1.0, 2.0, 0.8],
    [1.0, 0.5, 2.0],[1.0, 0.8, 1.8],
    [1.0, 0.3, 2.2],[1.0, -0.2, 1.8],
    [1.0, 1.2, 1.8],[1.0, 0.9, 1.6],
    [1.0, 1.8, 0.9],[1.0, 2.2, 0.7],
    [1.0, 2.5, 0.6],[1.0, 3.0, 1.0],
    [1.0, 1.6, 1.2],[1.0, 1.4, 1.3],
    [1.0, 0.6, 1.9],[1.0, 1.1, 1.4],
    [1.0, 2.3, 0.9], [1.0, 1.7, 1.1]
])

result1 = []
result2 = []

print("\n测试点分类结果:")
for point in test_points:
    classification = np.dot(a, point)
    class_str = "第一类" if classification > 0 else "第二类"
    print(f"测试点 ({point[1]:.2f}, {point[2]:.2f}) 被分类为 {class_str}")
    if classification > 0:
        result1.append(point[1:])
    else:
        result2.append(point[1:])

# 绘制测试点
if result1:
    result1 = np.array(result1).T
    plt.plot(result1[0], result1[1], 'g*', markersize=12, markeredgewidth=1, markeredgecolor='k', label="第一类测试点")

if result2:
    result2 = np.array(result2).T
    plt.plot(result2[0], result2[1], 'yo', markersize=12, markeredgewidth=1, markeredgecolor='k', label="第二类测试点")

# 添加分类区域背景
xx, yy = np.meshgrid(np.linspace(xmin, xmax, 200),
                     np.linspace(min(min(y1), min(y2)) - 1, max(max(y1), max(y2)) + 1, 200))
Z = a[0] + a[1] * xx + a[2] * yy
plt.contourf(xx, yy, Z > 0, alpha=0.1, levels=[0, 0.5, 1], colors=['blue', 'red'])

plt.title('2.2-徐文杰设计的增强测试数据的感知器算法分类图', fontsize=14)
plt.xlabel('X 特征', fontsize=12)
plt.ylabel('Y 特征', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper right', fontsize=10)
plt.axis([xmin, xmax, min(min(y1), min(y2)) - 1, max(max(y1), max(y2)) + 1])
plt.tight_layout()
plt.show()