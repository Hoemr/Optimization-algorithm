function [fitness,peisong] = fun(pop,distance,data,theta,beta)
%函数用于计算粒子适应度值
%pop           input           输入粒子 
%fitness       output          粒子适应度值 

r = 3000;   % 服务半径
rho = 1.2;    % 距离折算系数
F = 187500;    % 社区店建设成本
c = 0.007;    % 单位距离单位重量的配送费用
P = 65;    % 单位生鲜电商产品的平均价格
M = 50000;    % 社区店的建设容量  
v = 8.33;   % 速度    主要表示新鲜度随时间的变化部分, T = d/v
lambda = 4;   % 社区店管理运营成本与运营容量间的系数

g = 1;    % 随温度变化而变化的反应速率

peisong = zeros(2,length(pop));    % 第一行为每个小区的配送小区，第二行为每个供应小区目前已经供应的量
peisong(2,find(pop == 1)) = data(find(pop == 1),5)';   

%% ----------------确定每个小区的配送小区 -------------------%%
for i = 1:length(pop)   % 遍历每一个小区
    if pop(i)==1   % 如果是供应点
        peisong(1,i) = i;
    else
        % 如果i不是供应点，对i到j的距离进行排序
        [~,index] = sort(distance(i,:));
        for j = 2:length(index)   % 第一个数肯定是i，按照距离选择配送小区
            if pop(index(j)) == 1  % 一定要是配送点
                % 如果没有超出容量
                if peisong(2,index(j))+ data(i,5) <= M && distance(i,index(j)) <= r
                    peisong(1,i) = index(j);
                    peisong(2,index(j)) = peisong(2,index(j)) + data(i,5);
                    break
                end
            end
        end
    end
end

if length(find(peisong(1,:) == 0)) >= 1
    fitness = inf;
    return
end
%% --------------------计算总费用--------------------------%%
%%%%%%%%%%%%%%% 建设成本
C1 = sum(pop) * F;   % 供应小区个数乘以F

%%%%%%%%%%%%%%% 末端配送成本
C2 = 0;
for j = 1:length(pop)   % 遍历每一种配送情况,i其实也可以理解为第i个需求小区，也即终点
    C2 = C2 + c*distance(j,peisong(1,j))*data(j,5);
end
    
%%%%%%%%%%%%%%% 管理运营成本
C3 = 0;
for i = 1:length(pop)
    if pop(i) == 1
        C3 = C3 + lambda*P*M;
    end
end

%%%%%%%%%%%%%%% 新鲜度损失成本
C4 = 0;
for j = 1:length(pop)   % 遍历每一种配送情况
    C4 = C4 + P*data(j,5)*beta*(1 - g*exp(-theta*distance(j,peisong(1,j))/v));
end

%%%%%%%%%%%%%%% 总成本
C = C1+C2+C3+C4;

fitness = C;

