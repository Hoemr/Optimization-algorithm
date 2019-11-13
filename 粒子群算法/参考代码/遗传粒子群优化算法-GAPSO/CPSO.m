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
u=2;%混沌系数
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
  
    %%对粒子群最优位置进行混沌优化
      y(1,:)=(zbest-popmin)/(popmax-popmin);%将最优位置映射到Logistic方程的定义域[0,1]
      fitness(1)=fun(y(1,:)); 
        for t=1:sizepop-1 %通过Logistic方程进行M次迭代，得到混沌序列
            for e=1:D
        y(t+1,e)=u*y(t,e)*(1-y(t,e)); 
            end
        y(t+1,:)=popmin+(popmax-popmin)*y(t+1,:);%将混沌序列逆射到原解空间
        fitness(t+1)=fun(y(t+1,:)); %计算混沌变量可行解序列的适应度值
        end
[ybestfitness ybestindex]=min(fitness);%寻找最优混沌可行解矢量
  ybest=y(ybestindex,:);
        ran=1+fix(rand()*sizepop);%产生一随机数1~sizepop之间
        pop(ran,:)=ybest;
    yy(i)=fitnesszbest;    
        
end
%% 结果分析
plot(yy,'m','LineWidth',5)
title('多峰函数-Generaliaed Rastrigin最优个体适应度曲线','fontsize',20);
xlabel('迭代次数','fontsize',25);ylabel('适应度值','fontsize',25);
legend('基本粒子群算法','混沌粒子群算法','fontsize',30);
grid on
hold on
display('混沌粒子群算法输出结果');
zbest
minbest=min(yy)
meanbest=mean(yy)
stdbest=std(yy)