import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 设置中文字体
font_path = "C:/Windows/Fonts/msyh.ttc"
font_prop = font_manager.FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()

# 数据定义
x1 = [0.23, 1.52, 0.65, 0.77, 1.05, 1.19, 0.29, 0.25, 0.66, 0.56, 0.90, 0.13, -0.54, 0.94, -0.21,
      0.05, -0.08, 0.73, 0.33, 1.06, -0.02, 0.11, 0.31, 0.66]
y1 = [2.34, 2.19, 1.67, 1.63, 1.78, 2.01, 2.06, 2.12, 2.47, 1.51, 1.96, 1.83, 1.87, 2.29, 1.77,
      2.39, 1.56, 1.93, 2.20, 2.45, 1.75, 1.69, 2.48, 1.72]

x2 = [1.40, 1.23, 2.08, 1.16, 1.37, 1.18, 1.76, 1.97, 2.41, 2.58, 2.84, 1.95, 1.25, 1.28, 1.26,
      2.01, 2.18, 1.79, 1.33, 1.15, 1.70, 1.59, 2.93, 1.46]
y2 = [1.02, 0.96, 0.91, 1.49, 0.82, 0.93, 1.14, 1.06, 0.81, 1.28, 1.46, 1.43, 0.71, 1.29, 1.37,
      0.93, 1.22, 1.18, 0.87, 0.55, 0.51, 0.99, 0.91, 0.71]

# 数据预处理：增广向量形式
class1 = np.column_stack((np.ones(len(x1)), x1, y1, np.ones(len(x1))))  # 最后一列为类别标签
class2 = np.column_stack((np.ones(len(x2)), x2, y2, -np.ones(len(x2))))  # 最后一列为类别标签

# 合并数据并打乱顺序
data = np.vstack((class1, class2))
np.random.seed(0)  # 固定随机种子以便复现
np.random.shuffle(data)

# 初始化权重向量 a = [a0, a1, a2] = [1, 0, 0]
a = np.array([1.0, 0.0, 0.0])
learning_rate = 1.0
max_iter = 1000

# 感知器算法迭代
for iter in range(max_iter):
    misclassified = 0
    for sample in data:
        xi = sample[:3]  # 增广特征向量
        yi = sample[3]  # 类别标签

        if np.sign(np.dot(a, xi)) != yi:
            a += learning_rate * yi * xi
            misclassified += 1

    if misclassified == 0:
        print(f"收敛于第 {iter + 1} 次迭代")
        break
else:
    print(f"达到最大迭代次数 {max_iter}")

# 显示结果
print("\n最终权重向量: a =", a)
print("决策函数: d(x,y) = {:.4f} + {:.4f}*x + {:.4f}*y".format(a[0], a[1], a[2]))

# 测试点分类
test_points = np.array([[1, 1.5], [1.2, 1.0], [2.0, 0.9], [1.2, 1.5], [0.23, 2.33]])
print("\n测试点分类结果:")
for i, (x, y) in enumerate(test_points, 1):
    point = np.array([1, x, y])  # 增广特征向量
    pred = np.sign(np.dot(a, point))
    class_name = '类别1' if pred > 0 else '类别2'
    print(f"点 ({x:.2f}, {y:.2f}) 属于 {class_name}")

# 绘制结果
plt.figure(figsize=(10, 8))
plt.grid(True)

# 绘制原始数据点
plt.scatter(x1, y1, c='blue', label='类别1')
plt.scatter(x2, y2, c='red', label='类别2')

# 绘制决策边界
x_min, x_max = min(min(x1), min(x2)) - 0.5, max(max(x1), max(x2)) + 0.5
x_plot = np.linspace(x_min, x_max, 100)
y_plot = (-a[0] - a[1] * x_plot) / a[2]
plt.plot(x_plot, y_plot, 'k-', linewidth=2, label='决策边界')

# 绘制测试点
for i, (x, y) in enumerate(test_points, 1):
    plt.scatter(x, y, c='black', marker='x', s=100, linewidths=2)
    plt.text(x + 0.05, y + 0.05, str(i), fontsize=12)

plt.xlabel('x')
plt.ylabel('y')
plt.title('感知器算法分类结果')
plt.legend()
plt.axis('equal')
plt.show()