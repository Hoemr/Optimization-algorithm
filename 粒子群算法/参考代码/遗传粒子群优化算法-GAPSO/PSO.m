
%% 清空环境
clc
clear

%% 参数初始化
%粒子群算法中的两个参数
c1 = 1.49445;
c2 = 1.49445;
D=10;%粒子维数
maxgen=1000;   % 进化次数  
sizepop=20;   %种群规模
Vmax=1;
Vmin=-1;
popmax=5;
popmin=-5;
randdata1= xlsread('randdata1');
randdata2= xlsread('randdata2');
%% 产生初始粒子和速度
for i=1:sizepop
    %随机产生一个种群
    pop(i,:)=randdata1(i,:);    %初始种群
    V(i,:)=randdata2(i,:);  %初始化速度
    %计算适应度
    fitness(i)=fun(pop(i,:));   %粒子的适应值
end

%% 个体极值和群体极值
[bestfitness bestindex]=min(fitness);
zbest=pop(bestindex,:);   %全局最佳
gbest=pop;    %个体最佳
fitnessgbest=fitness;   %个体最佳适应度值
fitnesszbest=bestfitness;   %全局最佳适应度值

%% 迭代寻优
for i=1:maxgen
    
    for j=1:sizepop
        
        %速度更新
        V(j,:) = V(j,:) + c1*rand*(gbest(j,:) - pop(j,:)) + c2*rand*(zbest - pop(j,:));
        V(j,find(V(j,:)>Vmax))=Vmax;
        V(j,find(V(j,:)<Vmin))=Vmin;
        
        %种群更新
        pop(j,:)=pop(j,:)+V(j,:);
        pop(j,find(pop(j,:)>popmax))=popmax;
        pop(j,find(pop(j,:)<popmin))=popmin;
        
        %适应度值
        fitness(j)=fun(pop(j,:)); 
   
    end
    
    for j=1:sizepop
        
        %个体最优更新
        if fitness(j) < fitnessgbest(j)
            gbest(j,:) = pop(j,:);
            fitnessgbest(j) = fitness(j);
        end
        
        %群体最优更新
        if fitness(j) < fitnesszbest
            zbest = pop(j,:);
            fitnesszbest = fitness(j);
        end
    end 
    yy(i)=fitnesszbest;   
        
end
%% 结果分析
plot(yy,'r','LineWidth',5);
title('单峰函数Speher Model最优个体适应度曲线','fontsize',20);
xlabel('迭代次数','fontsize',25);ylabel('适应度值','fontsize',25);
legend('基本粒子群算法','fontsize',30);
grid on
hold on
display('基本粒子群算法输出结果');
zbest
minbest=min(yy)
meanbest=mean(yy)
stdbest=std(yy)
