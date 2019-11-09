clear,clc
%% 数据预处理
% 读取数据
x_train = xlsread('train_data.xlsx','Sheet1','A2:C21');
x_pred = xlsread('predict_data.xlsx','Sheet1','H3:J22');
% 数据归一化,映射x->y，归一化的结果为y(需求一)
[y_train(:,1),ps(1)] = mapminmax(x_train(:,1)');   % 入选品位归一化
[y_train(:,2),ps(2)] = mapminmax(x_train(:,2)');    % 精矿品位归一化
[y_train(:,3),ps(3)] = mapminmax(x_train(:,3)');    % 选矿比归一化

%选择训练样本个数
num_train = floor(0.7*size(y_train(:,1)));
%构造随机选择序列
choose = randperm(length(y_train(:,1)));   
train_data = y_train(choose(1:num_train),:);   % 所有的训练集
% gscatter(train_data(:,1),train_data(:,2),train_data(:,3));
output_train = train_data(:,end);   % 训练集的输出
input_train = train_data(:,1:2);

test_data = y_train(choose(num_train+1:end),:);   % 所有的测试集
output_test = test_data(:,end);    % 测试集输出的真实值
input_test = test_data(:,1:2);    % 测试集的输入

%% 采用交叉验证选择参数（需求二）
[bestacc,bestc,bestg] = SVMcg(output_train,input_train,-8,8,-8,8,3,0.2,0.2);

%% 使用选取的参数训练模型（需求三）
% 训练模型
options = ['-c ',num2str(bestc),' -g ',num2str(bestg),' -s 3 -p 0.01 -n 0.1 -t 0'];
model = libsvmtrain(output_train,input_train,options);

% ---------------------测试集------------------------%
% 看测试集的预测好坏
[py_test,~,~] = libsvmpredict(output_test,input_test,model);
% 测试集结果的反归一化
py_test_guiyi = mapminmax('reverse',py_test,ps(3));  % 预测结果的归一化
output_test_guiyi = mapminmax('reverse',output_test,ps(3));  % 测试集真实输出的归一化

% ------------------需要预测的数据------------------%
% 对需要预测的数据作归一化
y_pred(:,1) = mapminmax('apply',x_pred(:,1)',ps(1));   % 入选品位归一化
y_pred(:,2) = mapminmax('apply',x_pred(:,2)',ps(2));    % 精矿品位归一化
y_pred(:,3) = mapminmax('apply',x_pred(:,3)',ps(3));    % 选矿比归一化

% 预测数据
[py_pred,~,dec_value] = libsvmpredict(y_pred(:,3),y_pred(:,1:2),model);
% 对预测结果做反归一化
py_pred_guiyi = mapminmax('reverse',py_pred,ps(3));

% 预测结果绘图
figure;
plot(1:length(y_pred(:,3)),x_pred(:,3),'r-*','linewidth',1.1);  % 真实值
set(gca,'linewidth',1.4)
hold on;
plot(1:length(y_pred(:,3)),py_pred_guiyi,'m-.o','linewidth',1.1);   % 预测值

legend('真实值','多项式核函数预测值');
grid on;
title('多项式核函数预测情况','fontweight','bold')
xlabel('测试集样本编号','fontweight','bold')
ylabel('选矿比','fontweight','bold')
grid on

%% 以下为工具箱参数的详细说明
%% libsvm调用格式
% model = libsvmtrain(train_label,train_data,options);
% [predict_label,accuracy/mse,dec_value] = libsvmpredict(test_label,test_data,model);

%% options参数设置
% - s svm类型：SVM模型设置类型（默认值为0）
%     0：C - SVC
%     1:nu - SVC
%     2：one - class SVM
%     3: epsilon - SVR    这个是带有惩罚项的SVR模型
%     4: nu - SVR
% - t 核函数类型：核函数设置类型（默认值为2）
%     0:线性核函数 u'v
%     1:多项式核函数(r *u'v + coef0)^degree
%     2:RBF 核函数 exp( -r|u - v|^2)
%     3:sigmiod核函数 tanh(r * u'v + coef0)
% - d degree:核函数中的 degree 参数设置（针对多项式核函数，默认值为3）
% - g r(gama):g为核函数，核函数中的gama参数设置（针对多项式/sigmoid 核函数/RBF/，默认值为属性数目的倒数）
% - r coef0:核函数中的coef0参数设置(针对多项式/sigmoid核函数，默认值为0)
% - c cost(惩罚函数):设置 C - SVC,epsilon - SVR 和 nu - SVR的参数（默认值为1）
% - n nu:设置 nu-SVC ，one - class SVM 和 nu - SVR的参数
% - p epsilon:设置 epsilon - SVR 中损失函数的值（默认值为0.1）
% - m cachesize:设置 cache 内存大小，以 MB 为单位（默认值为100）
% - e eps:设置允许的终止阈值（默认值为0.001）
% - h shrinking：是否使用启发式，0或1（默认值为1）
% - wi weight:设置第几类的参数 C 为 weight * C（对于 C - SVC 中的 C，默认值为1）
% - v n:n - fold 交互检验模式，n为折数，必须大于等于2

%% svmtrain的输出model
% optimization finished, #iter = 505	%iter为迭代次数
% nu = 0.679187						%nu 是你选择的核函数类型的参数
% obj = -113.509601, rho = 0.051363   %obj为SVM文件转换为的二次规划求解得到的最小值
% nSV = 259, nBSV = 18				%nSV 为标准支持向量个数(0<a[i]<c)；nBSV为边界上的支持向量个数(a[i]=c)；
% Total nSV = 259						%nBSV为边界上的支持向量个数(a[i]=c)；对于两类来说，因为只有一个分类模型Total nSV = nSV，
%                                     %但是对于多类，这个是各个分类模型的nSV之和
% model = 
%   包含以下字段的 struct:			
% 
%     Parameters: [5×1 double]		%结构体变量，依次保存的是 -s -t -d -g -r等参数
%       nr_class: 2				    %分类的个数   
%        totalSV: 259				    %总的支持向量个数
%            rho: 0.0514				%b=-model.rho
%          Label: [2×1 double]
%     sv_indices: [259×1 double]
%          ProbA: []
%          ProbB: []
%            nSV: [2×1 double]	    %每一类的支持向量的个数
%        sv_coef: [259×1 double]	    %支持向量的系数
%            SVs: [259×13 double]	    %具体的支持向量，以稀疏矩阵的形式存储
% 其中：
%    w*x+b=0 中
%      w=model.SVs'*model.sv_coef
%      b=-model.rho
% w是高维空间中分类 超平面的法向量，b是常数项。

%% model.Parameters参数表
% Parameters =
%          0
%     2.0000
%     3.0000
%     2.8000
%          0
% ```c
% -  model.Paraments 参数意义从上到下依次为
% 
% ```c
% -s svm类型：SVM模型设置类型（默认值为0）
%  
% - t 核函数类型：核函数设置类型（默认值为2）
%  
% - d degree:核函数中的 degree 参数设置（针对多项式核函数，默认值为3）
%  
% - g r(gama):核函数中的gama参数设置（针对多项式/sigmoid 核函数/RBF/，默认值为属性数目的倒数）
%  
% - r coef0:核函数中的coef0参数设置(针对多项式/sigmoid核函数，默认值为0)

%% model.Label model.nr_class
% model.label 表示数据集中类别的标签都有什么
% model.nr_class 表示数据集职工有多少个类别

%% model.totalSV model.nSV
% model.total SV 代表总共的支持向量机的数目，这里一共259个。
% model.nSV 代表每类样本的支持向量的数目，model.nSV 所代表的顺序是和 model.label 相对应。 标签为1的样本118个，标签为-1的样本 141 个。

%% model.sv_coef model.SVs model.rho
% sv_coef: [259x1 double]
%     SVs: [259x13 double]<br>  model.rho = 0.0514
% 
% sv_coef，承装的是259个支持向量在决策函数中的系数；
% model.SVs 承装的是259个支持向量；
% model.rho = 0.0514   是决策函数中的常数项的相反数。

%% accuracy
% 返回的accuracy从上到下的意义依次是：
% 1、分类准确率，分类问题中用到的参数指标；
% 2、平均平方误差（ mean squared error,MSE），回归问题中用到的参数指标；
% 3、平方相关系数（ squared correlation coefficient ，R^2），回归问题中用到的参数指标。