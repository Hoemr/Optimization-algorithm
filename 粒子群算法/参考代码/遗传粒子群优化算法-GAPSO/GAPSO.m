
clc
clear
%杂交概率：Pc
%杂交池大小比例：Sp
%最大迭代次数：M
%问题的维数：D
%目标函数取最小值时的自变量值：xm
%目标函数的最小值：fv
%% 参数初始化
%粒子群算法中的两个参数
c1 = 1.49445;%学习因子
c2 = 1.49445;%学习因子
wmax=0.9;%惯性因子最大值
wmin=0.4;%惯性因子最小值
D=10;%粒子维数
pc=0.5;%杂交概率
maxgen=1000;   % 迭代次数  
sizepop=20;   %种群规模
pm=0.05;%变异概率
Vmax=1;
Vmin=-1;
popmax=3;
popmin=-3;
randdata1= xlsread('randdata1');
randdata2= xlsread('randdata2');
%% 产生初始粒子和速度
for i=1:sizepop
    %随机产生一个种群
    pop(i,:)=randdata1(1,:);    %初始化粒子位置
    V(i,:)=randdata2(1,:);  %初始化粒子速度
    %pop(i,:)=rands(1,D);    %初始种群
    %V(i,:)=rands(1,D);  %初始化速度
    fitness(i)=fun(pop(i,:));   %计算每个粒子的适应度值
end

%% 个体极值和群体极值
[bestfitness bestindex]=min(fitness);
zbest=pop(bestindex,:);   %全局最佳
gbest=pop;    %个体最佳
fitnessgbest=fitness;   %个体最佳适应度值
fitnesszbest=bestfitness;   %全局最佳适应度值

%% 迭代寻优
for k=1:maxgen
    for j=1:sizepop
       
   
        %速度更新
        V(j,:) =V(j,:) + c1*rand*(gbest(j,:) - pop(j,:)) + c2*rand*(zbest - pop(j,:));
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
          [sortbest sortindexbest]=sort(fitnessgbest);%按照适应度大小进行排序
            numPool=round(pc*sizepop);            %杂交池的大小,round为取整
            for i=1:numPool
                Poolx(i,:)=pop(sortindexbest(i),:);
                PoolVx(i,:)=V(sortindexbest(i),:);
            end
            for i=1:numPool  %选择要进行杂交的粒子
                seed1=floor(rand()*numPool)+1;
                seed2=floor(rand()*numPool)+1;
                while seed1==seed2
                seed1=floor(rand()*numPool)+1;
                seed2=floor(rand()*numPool)+1;
                end
                pb=rand;
                %子代的位置计算
                childx1(i,:)=pb*Poolx(seed1,:)+(1-pb)*Poolx(seed2,:);
                %子代的速度计算
                childv1(i,:)=(PoolVx(seed1,:)+PoolVx(seed2,:))*norm(PoolVx(seed1,:))./norm(PoolVx(seed1,:)+PoolVx(seed2,:));
                if fun(pop(i,:))>fun(childx1(i,:))
                   pop(i,:)=childx1(i,:); %子代的位置替换父代位置
                    V(i,:)=childv1(i,:); %子代的速度替换父代速度
                end
                
            end
            
        %%进行高斯变异
           mutationpool=round(pm*sizepop);
       for i=1:mutationpool  %选择要进行变异的粒子
           seed3=floor(rand()*mutationpool)+1;
           mutationchild(i,:)=pop(seed3,:)*(1+ randn);
           if fun(pop(i,:))>fun(mutationchild(i,:))
                 pop(i,:)=mutationchild(i,:); %子代的位置替换父代位置
           end
       end
       
      
        %计算杂交变异后的粒子适应度值以及进行更新
           for q=1:sizepop
                %适应度值
        fitness(q)=fun(pop(q,:)); 
        %个体最优更新
        if fitness(q) < fitnessgbest(q)
            gbest(q,:) = pop(q,:);
            fitnessgbest(q) = fitness(q);
        end
        
        %群体最优更新
        if fitness(q) < fitnesszbest
            zbest = pop(q,:);
            fitnesszbest = fitness(q);
        end
           end 
       yy(k)=fitnesszbest;
end
%% 结果分析
plot(yy,'k','LineWidth',5)
title('多峰函数Generaliaed Rosenbrock最优个体适应度曲线','fontsize',20);
xlabel('迭代次数','fontsize',25);ylabel('适应度值','fontsize',25);
legend('基本粒子群算法','混沌粒子群算法','基于遗传思想的粒子群算法','fontsize',30);
hold on
display('遗传思想与粒子群混合算法输出结果');
zbest
minbest=min(yy)
meanbest=mean(yy)
stdbest=std(yy)