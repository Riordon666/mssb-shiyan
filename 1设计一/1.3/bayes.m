% 训练集矩阵
x1 = [0, 0; 2, 1; 1, 0];
x2 = [-1, 1; -2, 0; -2, -1];
x3 = [0, -2; 0, -1; 1, -2];

% 计算均值、协方差矩阵和其逆矩阵
c1 = cov(x1); c2 = cov(x2); c3 = cov(x3); % c1, c2, c3 为协方差矩阵
t1 = diag(c1); t2 = diag(c2); t3 = diag(c3); % 对角化后的协方差矩阵
cc1 = inv(c1); cc2 = inv(c2); cc3 = inv(c3); % 协方差矩阵的逆
d1 = det(c1); d2 = det(c2); d3 = det(c3); % 协方差矩阵的行列式
u1 = mean(x1); u2 = mean(x2); u3 = mean(x3); % 各类的均值
u1 = u1'; u2 = u2'; u3 = u3'; % 转置均值向量

% 待分类的点
p = [-2; 2];

% 计算判别函数值
p1 = -0.5 * ((p - u1)' * cc1 * (p - u1) + log(d1));
p2 = -0.5 * ((p - u2)' * cc2 * (p - u2) + log(d2));
p3 = -0.5 * ((p - u3)' * cc3 * (p - u3) + log(d3));

% 进行分类
if p1 > p2
    if p1 > p3
        w = 1;
    else
        w = 3;
    end
elseif p2 > p3
    w = 2;
else
    w = 3;
end

% 输出分类结果
fprintf('x=(-2,2)属于第 %d 类\n', w);

% 第三步：协方差矩阵不相等的情况
G = str2sym('[x; y]'); % 使用 str2sym 而不是 sym
g1 = simplify(-0.5 * G' * cc1 * G + (cc1 * u1)' * G - 0.5 * (u1' * cc1 * u1 + log(d1)));
g2 = simplify(-0.5 * G' * cc2 * G + (cc2 * u2)' * G - 0.5 * (u2' * cc2 * u2 + log(d2)));
g3 = simplify(-0.5 * G' * cc3 * G + (cc3 * u3)' * G - 0.5 * (u3' * cc3 * u3 + log(d3)));

% 分界面
g12 = simplify(g1 - g2); % 1,2类的分界面
g23 = simplify(g2 - g3); % 2,3类的分界面
g31 = simplify(g3 - g1); % 3,1类的分界面

% 画出三类的分界面
h1 = ezplot(g12); hold on;
set(h1, 'LineWidth', 2, 'Color', 'red');
h2 = ezplot(g23); hold on;
set(h2, 'LineWidth', 2, 'Color', 'blue');
h3 = ezplot(g31); hold on;
set(h3, 'LineWidth', 2, 'Color', 'yellow');
legend('g12', 'g23', 'g31');

% 画出训练数据点
plot(x1(1, 1), x1(1, 2), 'or'); hold on;
plot(x1(2, 1), x1(2, 2), 'or'); hold on;
plot(x1(3, 1), x1(3, 2), 'or'); hold on;
plot(x2(1, 1), x2(1, 2), 'xb'); hold on;
plot(x2(2, 1), x2(2, 2), 'xb'); hold on;
plot(x2(3, 1), x2(3, 2), 'xb'); hold on;
plot(x3(1, 1), x3(1, 2), '*y'); hold on;
plot(x3(2, 1), x3(2, 2), '*y'); hold on;
plot(x3(3, 1), x3(3, 2), '*y'); hold on;

% 设置图形标题和标签
title('1.3-徐文杰设计的Bayes分类: 协方差矩阵不相等');
xlabel('x1'); ylabel('x2');
grid;
box;
hold off;

% 第四步：协方差矩阵相等的情况
c = c1 + c2 + c3;
cc = inv(c);
Q = str2sym('[x; y]'); % 使用 str2sym 而不是 sym
gq1 = simplify((cc * u1)' * Q - 0.5 * u1' * cc * u1);
gq2 = simplify((cc * u2)' * Q - 0.5 * u2' * cc * u2);
gq3 = simplify((cc * u3)' * Q - 0.5 * u3' * cc * u3);

% 分界面
gq12 = simplify(gq1 - gq2); % 1,2类的分界面
gq23 = simplify(gq2 - gq3); % 2,3类的分界面
gq31 = simplify(gq3 - gq1); % 3,1类的分界面

% 画出三类的分界面
h1 = ezplot(gq12); hold on;
set(h1, 'LineWidth', 2, 'Color', 'red');
h2 = ezplot(gq23); hold on;
set(h2, 'LineWidth', 2, 'Color', 'blue');
h3 = ezplot(gq31); hold on;
set(h3, 'LineWidth', 2, 'Color', 'yellow');
legend('gq12', 'gq23', 'gq31');

% 画出训练数据点
plot(x1(1, 1), x1(1, 2), 'or'); hold on;
plot(x1(2, 1), x1(2, 2), 'or'); hold on;
plot(x1(3, 1), x1(3, 2), 'or'); hold on;
plot(x2(1, 1), x2(1, 2), 'xb'); hold on;
plot(x2(2, 1), x2(2, 2), 'xb'); hold on;
plot(x2(3, 1), x2(3, 2), 'xb'); hold on;
plot(x3(1, 1), x3(1, 2), '*y'); hold on;
plot(x3(2, 1), x3(2, 2), '*y'); hold on;
plot(x3(3, 1), x3(3, 2), '*y'); hold on;

% 设置图形标题和标签
title('1.3-徐文杰设计的Bayes分类: 协方差矩阵相等');
xlabel('x1'); ylabel('x2');
grid;
axis([-3, 3, -3, 3]);
hold off;
