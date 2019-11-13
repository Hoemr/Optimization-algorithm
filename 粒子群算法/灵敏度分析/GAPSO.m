function [bestfitness,zbest,all_C] = GAPSO(distance,data,theta,beta)
%% ������ʼ��
D=size(distance,1);%����ά��
pc=0.5;%�ӽ�����
pm=0.05;%�������
maxgen=2000;   % ��������  
sizepop=200;   %��Ⱥ��ģ

pop = zeros(sizepop,D);
V = zeros(sizepop,D);

% theta = 0.005; 
% beta = 0.2;    % ���ʲ�Ʒ�����ʶ��½����������������ʧϵ��
%% ������ʼ���Ӻ��ٶ�
fitness = zeros(1,sizepop);
for i=1:sizepop
    %�������һ����Ⱥ
    pop(i,:)=round(rand(1,D));    %��ʼ��Ⱥ
    [fitness(i),~] = fun(pop(i,:),distance,data,theta,beta);   %����ÿ�����ӵ���Ӧ��ֵ
end

%�����ά����D
%Ŀ�꺯��ȡ��Сֵʱ���Ա���ֵ��pop
%Ŀ�꺯������Сֵ��fitness

%% ���弫ֵ��Ⱥ�弫ֵ
[bestfitness,bestindex]=min(fitness);
zbest=pop(bestindex,:);   %ȫ�����
gbest = pop;    % ÿһ���������
fitnessgbest = fitness;   % ÿһ�����������Ӧ��ֵ
fitnesszbest = bestfitness;   %ȫ�������Ӧ��ֵ

%% ����Ѱ��
for k=1:maxgen
    %% ��Ⱥ����
    for j = 1:sizepop
       %% ���죨�滻�ٶȸ��µı��ֳ�ʼ�ٶ���һ�����ݣ�
        if rand < pm
            % ѡ������20��λ�ñ���
            position = randperm(D);   % ��1:D���������������
            position = position(1:20);
            pop(j,position) = abs(pop(j,position) - 1);  % 1��Ϊ0,0��Ϊ1
        end
        
       %% �ӽ����滻׷����Ⱥ������������Ų��裩
        % ׷����Ⱥ����(zbest)
        pos = randperm(D);
        pos = pos(1:10);    % �����10��λ�ý���
        pop(j,pos) = zbest(pos);   % �����Ÿ��彻��
        
        % ���ָ�������(gbest(j,:))
        pos = randperm(D);
        pos = pos(1:10); 
        pop(j,pos) = gbest(j,pos); 
        
       %% �����ӽ�������������Ӧ��ֵ�Լ����и���
        % ��Ӧ��ֵ
        [fitness(j),~]=fun(pop(j,:),distance,data,theta,beta); 
        
        % �������Ÿ���
        if fitness(j) < fitnessgbest(j)
            gbest(j,:) = pop(j,:);
            fitnessgbest(j) = fitness(j);
        end
        
        % Ⱥ�����Ÿ���
        if fitness(j) < fitnesszbest
            zbest = pop(j,:);
            fitnesszbest = fitness(j);
        end
    end
    % ��¼ÿһ��������Ӧ��
    yy(k) = fitnesszbest;
    
    % �����ȥ��50������ֵ���ֲ��䣬�������ɸ���(����ģ���˻��˼��)
    if k > 200
        if length(yy(k-49:k) == yy(k)) == 50
            for i=1:sizepop
                %�������һ����Ⱥ
                new_pop(i,:)=round(rand(1,D));    %��ʼ��Ⱥ
                [new_fitness(i),~] = fun(new_pop(i,:),distance,data,theta,beta);   %����ÿ�����ӵ���Ӧ��ֵ
                
                % �������Ÿ���
                if new_fitness(i) < fitnessgbest(i)
                    gbest(i,:) = new_pop(i,:);
                    fitnessgbest(i) = new_fitness(i);
                end
                
                % Ⱥ�����Ÿ���
                if new_fitness(i) < fitnesszbest
                    zbest = new_pop(i,:);
                    fitnesszbest = new_fitness(i);
                end
            end
        end
    end
end

[~,~,all_C] = fun(zbest,distance,data,theta,beta);
% %% ----------------------�������
% % ������Ӧ��ֵ�仯���ͼ
% figure
% plot(yy,'k','LineWidth',1.1)
% title('����С��ѡȡ�������Ÿ�����Ӧ������','fontsize',10);
% xlabel('��������','fontsize',10,'fontweight','bold');
% ylabel('��Ӧ��ֵ','fontsize',10,'fontweight','bold');
% grid on 
% hold on
% set(gca,'linewidth',1.1)

% % �������
% disp('�Ŵ�˼��������Ⱥ����㷨������');
% zbest
% minbest = min(yy)

