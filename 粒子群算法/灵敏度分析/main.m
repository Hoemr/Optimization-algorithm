clc
clear

%% 读取数据
distance = xlsread('距离Dij.xlsx','距离','B2:BO67');
data = xlsread('距离Dij.xlsx','信息','B2:F67');

%% -------灵敏度分析---------------- %%
% 定值取值
theta = 0.005; 
beta = 0.2;    % 生鲜产品的新鲜度下降对消费者需求的损失系数

% theta取值为[0.005,0.020]，beta取值为[0.2,0.4]
theta_interval = linspace(0.005,0.020,20);   % 在0.005到0.020之间生成20个点，包含端点
beta_interval = linspace(0.2,0.4,20);   

% 对theta做灵敏度分析
theta_fitness = zeros(1,length(theta_interval));
theta_C = cell(1,length(theta_interval));
for i = 1:length(theta_interval)
    [bestfitness,~,all_C] = GAPSO(distance,data,theta_interval(i),beta);
    theta_fitness(i) = bestfitness;
    theta_C{i} = all_C;    % all_C 是一个向量
end
    
% 对beta做灵敏度分析
beta_fitness = zeros(1,length(beta_interval));
beta_C = cell(1,length(beta_interval));
for i = 1:length(beta_interval)
    [bestfitness,~,all_C] = GAPSO(distance,data,theta,beta_interval(i));
    beta_fitness(i) = bestfitness;
    beta_C{i} = all_C;
end

%% 将all_C进行分类
% theta
theta_C1 = zeros(1,length(theta_interval));
theta_C2 = zeros(1,length(theta_interval));
theta_C3 = zeros(1,length(theta_interval));
theta_C4 = zeros(1,length(theta_interval));
for i = 1:length(theta_interval)
    theta_C1(i) = theta_C{i}(2);
    theta_C2(i) = theta_C{i}(3);
    theta_C3(i) = theta_C{i}(4);
    theta_C4(i) = theta_C{i}(5);
end

% beta
beta_C1 = zeros(1,length(beta_interval));
beta_C2 = zeros(1,length(beta_interval));
beta_C3 = zeros(1,length(beta_interval));
beta_C4 = zeros(1,length(beta_interval));
for i = 1:length(beta_interval)
    beta_C1(i) = beta_C{i}(2);
    beta_C2(i) = beta_C{i}(3);
    beta_C3(i) = beta_C{i}(4);
    beta_C4(i) = beta_C{i}(5);
end

%% 绘制灵敏度分析图
% theta
figure
Q1 = semilogy(theta_interval,theta_fitness,theta_interval,theta_C1,theta_interval,theta_C2,theta_interval,theta_C3,theta_interval,theta_C4);
set(Q1,'linewidth',1.1);

xlabel('θ取值范围','fontweight','bold');
ylabel('费用(单位：元)','fontweight','bold');

legend('总成本','建设成本','末端配送成本','管理运营成本','新鲜度损失成本','Location','best')


grid on
set(gca,'linewidth',1.1)    % 设置坐标轴句柄属性
    
% beta
figure
Q2 = semilogy(beta_interval,beta_fitness,beta_interval,beta_C1,beta_interval,beta_C2,beta_interval,beta_C3,beta_interval,beta_C4);
set(Q2,'linewidth',1.1);

xlabel('β取值范围','fontweight','bold');
ylabel('费用(单位：元)','fontweight','bold');

legend('总成本','建设成本','末端配送成本','管理运营成本','新鲜度损失成本','Location','best')

grid on
set(gca,'linewidth',1.1)    % 设置坐标轴句柄属性
    


% %% 绘制布局图
% D=size(distance,1);%粒子维数
% [~,peisong] = fun(zbest,distance,data,theta,beta);
% 
% gyqk = zeros(D);  % 供应情况
% for i = 1: size(peisong,2)
%     gyqk(peisong(1,i),i) = data(i,5);
% end
% 
% % 找出供货小区与非供货小区
% gonghuoxiaoqu = find(zbest == 1);
% feigonghuo = find(zbest == 0);
% 
% % 绘制供货小区
% figure
% plot(data(gonghuoxiaoqu,1),data(gonghuoxiaoqu,2),'s','markersize',6,'markerfacecolor',[0.5,0.5,0.5],'markeredgecolor','b')
% hold on
% 
% % 绘制需求小区
% plot(data(feigonghuo,1),data(feigonghuo,2),'o','markersize',3,'markerfacecolor','g','markeredgecolor','g')
% hold on
% 
% % 绘制路线
% for i = 1:size(peisong,2)
%     line([data(peisong(1,i),1),data(i,1)],[data(peisong(1,i),2),data(i,2)],'color','green')
%     hold on
% end
% 
% % 参数调整
% axis([min(data(:,1))-500,max(data(:,1))+500,min(data(:,2))-500,max(data(:,2))+500])
% xlabel('横坐标(单位:m)','fontweight','bold')
% ylabel('纵坐标(单位:m)','fontweight','bold')
% set(gca,'linewidth',1.1)