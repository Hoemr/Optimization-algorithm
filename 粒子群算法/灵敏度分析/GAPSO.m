function [bestfitness,zbest,all_C] = GAPSO(distance,data,theta,beta)
%% 参数初始化
D=size(distance,1);%粒子维数
pc=0.5;%杂交概率
pm=0.05;%变异概率
maxgen=2000;   % 迭代次数  
sizepop=200;   %种群规模

pop = zeros(sizepop,D);
V = zeros(sizepop,D);

% theta = 0.005; 
% beta = 0.2;    % 生鲜产品的新鲜度下降对消费者需求的损失系数
%% 产生初始粒子和速度
fitness = zeros(1,sizepop);
for i=1:sizepop
    %随机产生一个种群
    pop(i,:)=round(rand(1,D));    %初始种群
    [fitness(i),~] = fun(pop(i,:),distance,data,theta,beta);   %计算每个粒子的适应度值
end

%问题的维数：D
%目标函数取最小值时的自变量值：pop
%目标函数的最小值：fitness

%% 个体极值和群体极值
[bestfitness,bestindex]=min(fitness);
zbest=pop(bestindex,:);   %全局最佳
gbest = pop;    % 每一个个体最佳
fitnessgbest = fitness;   % 每一个个体最佳适应度值
fitnesszbest = bestfitness;   %全局最佳适应度值

%% 迭代寻优
for k=1:maxgen
    %% 种群更新
    for j = 1:sizepop
       %% 变异（替换速度更新的保持初始速度那一块内容）
        if rand < pm
            % 选择任意20个位置变异
            position = randperm(D);   % 对1:D的数进行随机排列
            position = position(1:20);
            pop(j,position) = abs(pop(j,position) - 1);  % 1变为0,0变为1
        end
        
       %% 杂交（替换追随种群最优与个体最优步骤）
        % 追随种群最优(zbest)
        pos = randperm(D);
        pos = pos(1:10);    % 随机找10个位置交叉
        pop(j,pos) = zbest(pos);   % 与最优个体交叉
        
        % 保持个体最优(gbest(j,:))
        pos = randperm(D);
        pos = pos(1:10); 
        pop(j,pos) = gbest(j,pos); 
        
       %% 计算杂交变异后的粒子适应度值以及进行更新
        % 适应度值
        [fitness(j),~]=fun(pop(j,:),distance,data,theta,beta); 
        
        % 个体最优更新
        if fitness(j) < fitnessgbest(j)
            gbest(j,:) = pop(j,:);
            fitnessgbest(j) = fitness(j);
        end
        
        % 群体最优更新
        if fitness(j) < fitnesszbest
            zbest = pop(j,:);
            fitnesszbest = fitness(j);
        end
    end
    % 记录每一代最优适应度
    yy(k) = fitnesszbest;
    
    % 如果过去的50代最优值保持不变，则新生成个体(引入模拟退火的思想)
    if k > 200
        if length(yy(k-49:k) == yy(k)) == 50
            for i=1:sizepop
                %随机产生一个种群
                new_pop(i,:)=round(rand(1,D));    %初始种群
                [new_fitness(i),~] = fun(new_pop(i,:),distance,data,theta,beta);   %计算每个粒子的适应度值
                
                % 个体最优更新
                if new_fitness(i) < fitnessgbest(i)
                    gbest(i,:) = new_pop(i,:);
                    fitnessgbest(i) = new_fitness(i);
                end
                
                % 群体最优更新
                if new_fitness(i) < fitnesszbest
                    zbest = new_pop(i,:);
                    fitnesszbest = new_fitness(i);
                end
            end
        end
    end
end

[~,~,all_C] = fun(zbest,distance,data,theta,beta);
% %% ----------------------结果分析
% % 绘制适应度值变化情况图
% figure
% plot(yy,'k','LineWidth',1.1)
% title('供货小区选取问题最优个体适应度曲线','fontsize',10);
% xlabel('迭代次数','fontsize',10,'fontweight','bold');
% ylabel('适应度值','fontsize',10,'fontweight','bold');
% grid on 
% hold on
% set(gca,'linewidth',1.1)

% % 结果分析
% disp('遗传思想与粒子群混合算法输出结果');
% zbest
% minbest = min(yy)

