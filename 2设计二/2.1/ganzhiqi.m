clc;
clear;
close all;

% 样本数据
x1 = [0.23 1.52 0.65 0.77 1.05 1.19 0.29 0.25 0.66 0.56 0.90 0.13 -0.54 0.94 -0.21 0.05 -0.08 0.73 0.33 1.06 -0.02 0.11 0.31 0.66];
y1 = [2.34 2.19 1.67 1.63 1.78 2.01 2.06 2.12 2.47 1.51 1.96 1.83 1.87 2.29 1.77 2.39 1.56 1.93 2.20 2.45 1.75 1.69 2.48 1.72];
x2 = [1.40 1.23 2.08 1.16 1.37 1.18 1.76 1.97 2.41 2.58 2.84 1.95 1.25 1.28 1.26 2.01 2.18 1.79 1.33 1.15 1.70 1.59 2.93 1.46];
y2 = [1.02 0.96 0.91 1.49 0.82 0.93 1.14 1.06 0.81 1.28 1.46 1.43 0.71 1.29 1.37 0.93 1.22 1.18 0.87 0.55 0.51 0.99 0.91 0.71];

% 初始化变量
w1 = zeros(2, 24);
w2 = zeros(2, 24);
for i = 1:24
    w1(:, i) = [x1(i); y1(i)];
    w2(:, i) = [x2(i); y2(i)];
end

% 增广矩阵
ww1 = [ones(1, 24); w1];
ww2 = [ones(1, 24); w2];

% 样本规范化
w12 = ww1 - ww2;

% 初始化权向量
a = [1; 1; 1];

% 批处理算法
y = zeros(1, size(w12, 2));
k = 0;
max_iter = 1000;  % 最大迭代次数
converged = false;

while ~converged && k < max_iter
    sum_err = zeros(3, 1);
    misclassified = 0;
    
    for i = 1:size(y, 2)
        y(i) = a' * w12(:, i);
        if y(i) <= 0
            sum_err = sum_err + w12(:, i);
            misclassified = misclassified + 1;
        end
    end
    
    if misclassified == 0
        converged = true;
    else
        a = a + sum_err;
        k = k + 1;
        % 显示迭代信息
        disp(['迭代: ', num2str(k), ' 错误分类数: ', num2str(misclassified)]);
    end
end

% 输出结果
disp('==============================');
if converged
    disp(['算法收敛于 ', num2str(k), ' 次迭代']);
else
    disp(['达到最大迭代次数 ', num2str(max_iter), ' 仍未收敛']);
end
disp('权向量 a:');
disp(a);

% 绘图
figure;
plot(w1(1, :), w1(2, :), '+r', 'MarkerSize', 8, 'LineWidth', 1.5);
hold on;
plot(w2(1, :), w2(2, :), 'bs', 'MarkerSize', 8, 'LineWidth', 1.5);

% 分界面计算
xmin = min([w1(1, :), w2(1, :)]) - 0.5;
xmax = max([w1(1, :), w2(1, :)]) + 0.5;
xindex = linspace(xmin, xmax, 100);
yindex = (-a(2)/a(3)) * xindex + (-a(1)/a(3));
plot(xindex, yindex, '-k', 'LineWidth', 2);

% 测试新数据点
test_points = [1 1.2 2.0 1.2 0.23; 
               1.5 1.0 0.9 1.5 2.33];
test_points_aug = [ones(1, 5); test_points];

result1 = [];
result2 = [];
for i = 1:5
    if a' * test_points_aug(:, i) > 0
        result1 = [result1, test_points(:, i)];
    else
        result2 = [result2, test_points(:, i)];
    end
end

% 绘制测试点
plot(result1(1, :), result1(2, :), '*k', 'MarkerSize', 10, 'LineWidth', 1.5);
plot(result2(1, :), result2(2, :), 'rd', 'MarkerSize', 10, 'LineWidth', 1.5);

% 图形美化
grid on;
title('2.1-徐文杰设计的感知器分类结果', 'FontSize', 14);
xlabel('特征x', 'FontSize', 12);
ylabel('特征y', 'FontSize', 12);
legend('类别1样本', '类别2样本', '决策边界', '测试点-类别1', '测试点-类别2', 'Location', 'best');
axis([xmin xmax min([w1(2, :), w2(2, :)])-0.5 max([w1(2, :), w2(2, :)])+0.5]);

% 显示分类结果
disp('测试点分类结果:');
disp('属于第一类的点:');
disp(result1);
disp('属于第二类的点:');
disp(result2);