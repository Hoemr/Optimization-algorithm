%遗传算法VRP问题求解
%%            变量说明和场景描述
%% 输入数据：
% D：节点距离矩阵
% NIND： 种群规模
% X：节点坐标
% MAXGEN：最大迭代次数
% max_vc:最大车辆容量
% Demand:节点需求矩阵
% max_d:车辆运输最大距离
% v:车辆速度
% cv:车辆运输单价
% max_v:可供调配的最大车辆数
%% 优化目标：运输成本最小 ，车辆数最少
%% 约束条件：满足车辆容量限制，满足车辆运输距离限制，同一节点只接收一辆车服务,
%%          单个节点位置的需求小于车辆的最大载重               
% clear
% clc
%%               初始化
% step1 加载车辆和服务信息数据
load zuobiao_X   %载入节点坐标:【节点编号，X,Y】（包含始节点）
% 计算结点距离矩阵
for i = 1:size(X,1)
    for j = 1:size(X,1)
        D(i,j) = sqrt((X(i,2)-X(j,2))^2+(X(i,3)-X(j,3))^2);
    end
end
N=size(X,1)-1;   %配送节点个数
gj_num = 3;   % 运输工具类型数量
load demand      %载入需求矩阵：【节点编号，需求量】（包含始节点）
max_vc = [50,27,13];  %运输工具最大载重
cv = [30,20,10];   %单位运输成本
cb = [200,150,100];   % 固定出车成本
%max_d=40;  %单车行驶最大距离
max_v=[6,3,2];   %假设可调配车辆数的最大值
best_obj_list = [];

% step2 GA参数初始化
NIND=150;  %种群大小
MAXGEN=20;   %最大迭代次数
Pc=0.9;   %交叉概率
Pm=0.05;   %变异概率
GGAP=0.9;  %代沟
%%               开始迭代
% step1 舒适化种群
chrom=initpop(NIND,N+gj_num);
gen=0;
% 迭代过程图
figure
hold on
box on;
xlim([0,MAXGEN])% x坐标轴上下限 x limit
title('优化过程')
xlabel('代数')
ylabel('最优值')

% 计算第一代个体的适应度，并找出最优个体及对应的解与染色体
[all_fitness,best_fitness,all_obj,best_obj,best_chrom,best_car_path] = fitness2(chrom,demand,N,max_vc,max_v,cb,cv,D);
best_obj_list = [best_obj_list best_obj];
pre_obj = best_obj;   % 上一代的最优目标函数值
for  gen = 1:MAXGEN
    %% 选择
    FitnV = all_fitness;
    SelCh=Select(chrom,FitnV,GGAP);
    %% 交叉操作
    SelCh=Recombin(SelCh,Pc);
    %% 变异
    SelCh=Mutate(SelCh,Pm);
    %% 重插入子代的新种群
    ObjV = all_obj;
    chrom=Reins(chrom,SelCh,ObjV);  % 合并成新种群
    %% 计算新种群的适应度等
    [all_fitness,best_fitness,all_obj,best_obj,best_chrom,best_car_path] = fitness2(chrom,demand,N,max_vc,max_v,cb,cv,D);
    best_obj_list = [best_obj_list best_obj];
    %% 绘图
    line([gen-1,gen],[pre_obj,best_obj]);
    pause(0.0001)
    pre_obj = best_obj;   % 上一代的最优目标函数值
end
%% 结果分析
% 最优目标函数值
disp(['最优目标函数值:',num2str(best_obj_list(end))])
% 最优时所用的交通工具数量（即路径数量）
car_num = size(best_car_path,1);
disp(['最优时总的交通工具使用量为：',num2str(car_num)])
