import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.datasets import make_blobs
from matplotlib import font_manager

# 设置中文字体
font_path = "C:/Windows/Fonts/msyh.ttc"
font_prop = font_manager.FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()
# 1. 生成可分线性数据（二维示例）
np.random.seed(1)  # 设置随机种子保证可重复性

# 生成两类数据（修正了center_box参数）
X, y = make_blobs(n_samples=100, centers=2, cluster_std=0.8,
                 center_box=(1.5, 3.5), random_state=1)
y = np.where(y == 0, -1, 1)  # 将标签转换为-1和1

# 可视化原始数据
plt.figure(figsize=(10, 6))
plt.scatter(X[y == 1, 0], X[y == 1, 1], c='b', marker='o', label='类别1')
plt.scatter(X[y == -1, 0], X[y == -1, 1], c='r', marker='+', label='类别2')
plt.title('原始数据分布')
plt.xlabel('特征1')
plt.ylabel('特征2')
plt.legend()
plt.grid(True)
plt.show()

# 2. 训练线性SVM模型
clf = svm.SVC(kernel='linear', C=1.0)
clf.fit(X, y)

# 获取模型参数
w = clf.coef_[0]  # 权重向量
b = clf.intercept_[0]  # 偏置项
sv = clf.support_vectors_  # 支持向量

print('SVM模型参数:')
print(f'权重向量 w = [{w[0]:.4f}, {w[1]:.4f}]')
print(f'偏置项 b = {b:.4f}')
print(f'支持向量数量: {len(sv)}')

# 3. 可视化决策边界和支持向量
plt.figure(figsize=(10, 6))

# 绘制数据点
plt.scatter(X[y == 1, 0], X[y == 1, 1], c='b', marker='o', label='类别1')
plt.scatter(X[y == -1, 0], X[y == -1, 1], c='r', marker='+', label='类别2')

# 绘制支持向量
plt.scatter(sv[:, 0], sv[:, 1], s=100, facecolors='none', edgecolors='k',
            linewidths=1.5, label='支持向量')

# 绘制决策边界
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# 创建网格评估模型
xx = np.linspace(xlim[0], xlim[1], 100)
yy = np.linspace(ylim[0], ylim[1], 100)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)

# 绘制决策边界和间隔
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])
ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100,
           linewidth=1, facecolors='none', edgecolors='k')

plt.title('线性SVM分类结果')
plt.xlabel('特征1')
plt.ylabel('特征2')
plt.legend()
plt.grid(True)
plt.show()

# 4. 模型评估
y_pred = clf.predict(X)
accuracy = accuracy_score(y, y_pred)
print(f'\n模型准确率: {accuracy:.2%}')

print('混淆矩阵:')
print(confusion_matrix(y, y_pred))

# 5. 新样本预测示例
new_X = np.array([[2.5, 3.0],
                 [1.8, 2.2],
                 [3.2, 3.5]])
new_y_pred = clf.predict(new_X)

print('\n新样本预测结果:')
for i, (sample, pred) in enumerate(zip(new_X, new_y_pred)):
    print(f'样本 {sample} -> 预测类别: {pred}')