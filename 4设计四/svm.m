%% 线性SVM分类案例 - MATLAB实现
clc; clear; close all;

%% 1. 生成可分线性数据（二维示例）
rng(1); % 设置随机种子保证可重复性

% 类别1数据
n1 = 50;
X1 = 1.5 + 0.8*randn(n1,2);

% 类别2数据
n2 = 50;
X2 = 3.5 + 0.8*randn(n2,2);

% 合并数据
X = [X1; X2];
y = [ones(n1,1); -ones(n2,1)];

% 可视化原始数据
figure;
gscatter(X(:,1), X(:,2), y, 'rb', 'o+');
title('原始数据分布');
xlabel('特征1'); ylabel('特征2');
legend('类别1', '类别2');
grid on;

%% 2. 训练线性SVM模型
svmModel = fitcsvm(X, y, 'KernelFunction', 'linear', ...
    'BoxConstraint', 1, 'Standardize', true);

% 获取模型参数
w = svmModel.Beta;  % 权重向量
b = svmModel.Bias;  % 偏置项
sv = svmModel.SupportVectors; % 支持向量

fprintf('SVM模型参数:\n');
fprintf('权重向量 w = [%.4f, %.4f]\n', w(1), w(2));
fprintf('偏置项 b = %.4f\n', b);
fprintf('支持向量数量: %d\n', size(sv,1));

%% 3. 可视化决策边界和支持向量
figure;
% 绘制数据点
gscatter(X(:,1), X(:,2), y, 'rb', 'o+');
hold on;

% 绘制支持向量
plot(sv(:,1), sv(:,2), 'ko', 'MarkerSize', 10, 'LineWidth', 1.5);

% 绘制决策边界
x1 = linspace(min(X(:,1))-0.5, max(X(:,1))+0.5, 100);
x2 = -(w(1)*x1 + b)/w(2);
plot(x1, x2, 'k-', 'LineWidth', 2);

% 绘制间隔边界
plot(x1, -(w(1)*x1 + b + 1)/w(2), 'k--');
plot(x1, -(w(1)*x1 + b - 1)/w(2), 'k--');

title('线性SVM分类结果');
xlabel('特征1'); ylabel('特征2');
legend('类别1', '类别2', '支持向量', '决策边界', '间隔边界');
grid on;
hold off;

%% 4. 模型评估
% 预测训练数据
yPred = predict(svmModel, X);

% 计算准确率
accuracy = sum(yPred == y)/length(y);
fprintf('\n模型准确率: %.2f%%\n', accuracy*100);

% 混淆矩阵
confMat = confusionmat(y, yPred);
disp('混淆矩阵:');
disp(confMat);

%% 5. 新样本预测示例
newX = [2.5, 3.0;
        1.8, 2.2;
        3.2, 3.5];
newYPred = predict(svmModel, newX);

fprintf('\n新样本预测结果:\n');
for i = 1:size(newX,1)
    fprintf('样本 [%.2f, %.2f] -> 预测类别: %d\n', ...
            newX(i,1), newX(i,2), newYPred(i));
end