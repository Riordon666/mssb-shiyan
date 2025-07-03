clc;
clear;
close all;

% 数据定义
x1 = [0.23, 1.52, 0.65, 0.77, 1.05, 1.19, 0.29, 0.25, 0.66, 0.56, 0.90, 0.13, -0.54, 0.94, -0.21, 0.05, -0.08, 0.73, 0.33, 1.06, -0.02, 0.11, 0.31, 0.66];
y1 = [2.34, 2.19, 1.67, 1.63, 1.78, 2.01, 2.06, 2.12, 2.47, 1.51, 1.96, 1.83, 1.87, 2.29, 1.77, 2.39, 1.56, 1.93, 2.20, 2.45, 1.75, 1.69, 2.48, 1.72];

x2 = [1.40, 1.23, 2.08, 1.16, 1.37, 1.18, 1.76, 1.97, 2.41, 2.58, 2.84, 1.95, 1.25, 1.28, 1.26, 2.01, 2.18, 1.79, 1.33, 1.15, 1.70, 1.59, 2.93, 1.46];
y2 = [1.02, 0.96, 0.91, 1.49, 0.82, 0.93, 1.14, 1.06, 0.81, 1.28, 1.46, 1.43, 0.71, 1.29, 1.37, 0.93, 1.22, 1.18, 0.87, 0.55, 0.51, 0.99, 0.91, 0.71];

% 数据预处理：增广向量形式，类别1标签为1，类别2标签为-1
class1 = [ones(1, length(x1)); x1; y1; ones(1, length(x1))]; % 最后一列为类别标签
class2 = [ones(1, length(x2)); x2; y2; -ones(1, length(x2))]; % 最后一列为类别标签

% 合并数据并打乱顺序
data = [class1, class2]';
rng(0); % 固定随机种子以便复现
data = data(randperm(size(data, 1)), :);

% 初始化权重向量 a = [a0, a1, a2] = [1, 0, 0]
a = [1, 0, 0];
learning_rate = 1;
max_iter = 1000;
iter = 0;
converged = false;

% 感知器算法迭代
for iter = 1:max_iter
    misclassified = 0;
    for i = 1:size(data, 1)
        xi = data(i, 1:3)'; % 增广特征向量
        yi = data(i, 4);    % 类别标签

        if sign(a * xi) ~= yi
            a = a + learning_rate * yi * xi';
            misclassified = misclassified + 1;
        end
    end

    if misclassified == 0
        converged = true;
        break;
    end
end

% 显示结果
fprintf('迭代次数: %d\n', iter);
fprintf('最终权重向量: a = [%.4f, %.4f, %.4f]\n', a(1), a(2), a(3));
fprintf('决策函数: d(x,y) = %.4f + %.4f*x + %.4f*y\n', a(1), a(2), a(3));

% 测试点分类
test_points = [1, 1.5; 1.2, 1.0; 2.0, 0.9; 1.2, 1.5; 0.23, 2.33];
fprintf('\n测试点分类结果:\n');
for i = 1:size(test_points, 1)
    point = [1, test_points(i, 1), test_points(i, 2)]; % 增广特征向量
    pred = sign(a * point');
    if pred > 0
        class = '类别1';
    else
        class = '类别2';
    end
    fprintf('点 (%.2f, %.2f) 属于 %s\n', test_points(i,1), test_points(i,2), class);
end

% 绘制结果
figure;
hold on;
grid on;

% 绘制原始数据点
scatter(x1, y1, 'filled', 'DisplayName', '类别1');
scatter(x2, y2, 'filled', 'DisplayName', '类别2');

% 绘制决策边界
x_plot = linspace(min([x1,x2])-0.5, max([x1,x2])+0.5, 100);
y_plot = (-a(1) - a(2)*x_plot)/a(3);
plot(x_plot, y_plot, 'k-', 'LineWidth', 2, 'DisplayName', '决策边界');

% 绘制测试点
scatter(test_points(:,1), test_points(:,2), 100, 'k', 'x', 'LineWidth', 2, 'DisplayName', '测试点');

% 标注测试点
for i = 1:size(test_points, 1)
    text(test_points(i,1)+0.05, test_points(i,2)+0.05, num2str(i), 'FontSize', 10);
end

xlabel('x');
ylabel('y');
title('感知器算法分类结果');
legend('Location', 'best');
axis equal;
hold off;