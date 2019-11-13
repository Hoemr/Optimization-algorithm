
%% ��ջ���
clc
clear

%% ������ʼ��
%����Ⱥ�㷨�е���������
c1 = 1.49445;
c2 = 1.49445;
D=10;%����ά��
maxgen=1000;   % ��������  
sizepop=20;   %��Ⱥ��ģ
Vmax=1;
Vmin=-1;
popmax=5;
popmin=-5;
randdata1= xlsread('randdata1');
randdata2= xlsread('randdata2');
%% ������ʼ���Ӻ��ٶ�
for i=1:sizepop
    %�������һ����Ⱥ
    pop(i,:)=randdata1(i,:);    %��ʼ��Ⱥ
    V(i,:)=randdata2(i,:);  %��ʼ���ٶ�
    %������Ӧ��
    fitness(i)=fun(pop(i,:));   %���ӵ���Ӧֵ
end

%% ���弫ֵ��Ⱥ�弫ֵ
[bestfitness bestindex]=min(fitness);
zbest=pop(bestindex,:);   %ȫ�����
gbest=pop;    %�������
fitnessgbest=fitness;   %���������Ӧ��ֵ
fitnesszbest=bestfitness;   %ȫ�������Ӧ��ֵ

%% ����Ѱ��
for i=1:maxgen
    
    for j=1:sizepop
        
        %�ٶȸ���
        V(j,:) = V(j,:) + c1*rand*(gbest(j,:) - pop(j,:)) + c2*rand*(zbest - pop(j,:));
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
    yy(i)=fitnesszbest;   
        
end
%% �������
plot(yy,'r','LineWidth',5);
title('���庯��Speher Model���Ÿ�����Ӧ������','fontsize',20);
xlabel('��������','fontsize',25);ylabel('��Ӧ��ֵ','fontsize',25);
legend('��������Ⱥ�㷨','fontsize',30);
grid on
hold on
display('��������Ⱥ�㷨������');
zbest
minbest=min(yy)
meanbest=mean(yy)
stdbest=std(yy)
