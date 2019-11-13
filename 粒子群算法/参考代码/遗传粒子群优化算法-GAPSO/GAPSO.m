
clc
clear
%�ӽ����ʣ�Pc
%�ӽ��ش�С������Sp
%������������M
%�����ά����D
%Ŀ�꺯��ȡ��Сֵʱ���Ա���ֵ��xm
%Ŀ�꺯������Сֵ��fv
%% ������ʼ��
%����Ⱥ�㷨�е���������
c1 = 1.49445;%ѧϰ����
c2 = 1.49445;%ѧϰ����
wmax=0.9;%�����������ֵ
wmin=0.4;%����������Сֵ
D=10;%����ά��
pc=0.5;%�ӽ�����
maxgen=1000;   % ��������  
sizepop=20;   %��Ⱥ��ģ
pm=0.05;%�������
Vmax=1;
Vmin=-1;
popmax=3;
popmin=-3;
randdata1= xlsread('randdata1');
randdata2= xlsread('randdata2');
%% ������ʼ���Ӻ��ٶ�
for i=1:sizepop
    %�������һ����Ⱥ
    pop(i,:)=randdata1(1,:);    %��ʼ������λ��
    V(i,:)=randdata2(1,:);  %��ʼ�������ٶ�
    %pop(i,:)=rands(1,D);    %��ʼ��Ⱥ
    %V(i,:)=rands(1,D);  %��ʼ���ٶ�
    fitness(i)=fun(pop(i,:));   %����ÿ�����ӵ���Ӧ��ֵ
end

%% ���弫ֵ��Ⱥ�弫ֵ
[bestfitness bestindex]=min(fitness);
zbest=pop(bestindex,:);   %ȫ�����
gbest=pop;    %�������
fitnessgbest=fitness;   %���������Ӧ��ֵ
fitnesszbest=bestfitness;   %ȫ�������Ӧ��ֵ

%% ����Ѱ��
for k=1:maxgen
    for j=1:sizepop
       
   
        %�ٶȸ���
        V(j,:) =V(j,:) + c1*rand*(gbest(j,:) - pop(j,:)) + c2*rand*(zbest - pop(j,:));
        V(j,find(V(j,:)>Vmax))=Vmax;
        V(j,find(V(j,:)<Vmin))=Vmin;
        
        %��Ⱥ����
        pop(j,:)=pop(j,:)+V(j,:);
        pop(j,find(pop(j,:)>popmax))=popmax;
        pop(j,find(pop(j,:)<popmin))=popmin;
        %��Ӧ��ֵ
        fitness(j)=fun(pop(j,:)); 
   
    end
     
    for j=1:sizepop
        
        %�������Ÿ���
        if fitness(j) < fitnessgbest(j)
            gbest(j,:) = pop(j,:);
            fitnessgbest(j) = fitness(j);
        end
        
        %Ⱥ�����Ÿ���
        if fitness(j) < fitnesszbest
            zbest = pop(j,:);
            fitnesszbest = fitness(j);
        end
    end 
          [sortbest sortindexbest]=sort(fitnessgbest);%������Ӧ�ȴ�С��������
            numPool=round(pc*sizepop);            %�ӽ��صĴ�С,roundΪȡ��
            for i=1:numPool
                Poolx(i,:)=pop(sortindexbest(i),:);
                PoolVx(i,:)=V(sortindexbest(i),:);
            end
            for i=1:numPool  %ѡ��Ҫ�����ӽ�������
                seed1=floor(rand()*numPool)+1;
                seed2=floor(rand()*numPool)+1;
                while seed1==seed2
                seed1=floor(rand()*numPool)+1;
                seed2=floor(rand()*numPool)+1;
                end
                pb=rand;
                %�Ӵ���λ�ü���
                childx1(i,:)=pb*Poolx(seed1,:)+(1-pb)*Poolx(seed2,:);
                %�Ӵ����ٶȼ���
                childv1(i,:)=(PoolVx(seed1,:)+PoolVx(seed2,:))*norm(PoolVx(seed1,:))./norm(PoolVx(seed1,:)+PoolVx(seed2,:));
                if fun(pop(i,:))>fun(childx1(i,:))
                   pop(i,:)=childx1(i,:); %�Ӵ���λ���滻����λ��
                    V(i,:)=childv1(i,:); %�Ӵ����ٶ��滻�����ٶ�
                end
                
            end
            
        %%���и�˹����
           mutationpool=round(pm*sizepop);
       for i=1:mutationpool  %ѡ��Ҫ���б��������
           seed3=floor(rand()*mutationpool)+1;
           mutationchild(i,:)=pop(seed3,:)*(1+ randn);
           if fun(pop(i,:))>fun(mutationchild(i,:))
                 pop(i,:)=mutationchild(i,:); %�Ӵ���λ���滻����λ��
           end
       end
       
      
        %�����ӽ�������������Ӧ��ֵ�Լ����и���
           for q=1:sizepop
                %��Ӧ��ֵ
        fitness(q)=fun(pop(q,:)); 
        %�������Ÿ���
        if fitness(q) < fitnessgbest(q)
            gbest(q,:) = pop(q,:);
            fitnessgbest(q) = fitness(q);
        end
        
        %Ⱥ�����Ÿ���
        if fitness(q) < fitnesszbest
            zbest = pop(q,:);
            fitnesszbest = fitness(q);
        end
           end 
       yy(k)=fitnesszbest;
end
%% �������
plot(yy,'k','LineWidth',5)
title('��庯��Generaliaed Rosenbrock���Ÿ�����Ӧ������','fontsize',20);
xlabel('��������','fontsize',25);ylabel('��Ӧ��ֵ','fontsize',25);
legend('��������Ⱥ�㷨','��������Ⱥ�㷨','�����Ŵ�˼�������Ⱥ�㷨','fontsize',30);
hold on
display('�Ŵ�˼��������Ⱥ����㷨������');
zbest
minbest=min(yy)
meanbest=mean(yy)
stdbest=std(yy)