clc
clear

%% ��ȡ����
distance = xlsread('����Dij.xlsx','����','B2:BO67');
data = xlsread('����Dij.xlsx','��Ϣ','B2:F67');

%% -------�����ȷ���---------------- %%
% ��ֵȡֵ
theta = 0.005; 
beta = 0.2;    % ���ʲ�Ʒ�����ʶ��½����������������ʧϵ��

% thetaȡֵΪ[0.005,0.020]��betaȡֵΪ[0.2,0.4]
theta_interval = linspace(0.005,0.020,20);   % ��0.005��0.020֮������20���㣬�����˵�
beta_interval = linspace(0.2,0.4,20);   

% ��theta�������ȷ���
theta_fitness = zeros(1,length(theta_interval));
theta_C = cell(1,length(theta_interval));
for i = 1:length(theta_interval)
    [bestfitness,~,all_C] = GAPSO(distance,data,theta_interval(i),beta);
    theta_fitness(i) = bestfitness;
    theta_C{i} = all_C;    % all_C ��һ������
end
    
% ��beta�������ȷ���
beta_fitness = zeros(1,length(beta_interval));
beta_C = cell(1,length(beta_interval));
for i = 1:length(beta_interval)
    [bestfitness,~,all_C] = GAPSO(distance,data,theta,beta_interval(i));
    beta_fitness(i) = bestfitness;
    beta_C{i} = all_C;
end

%% ��all_C���з���
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

%% ���������ȷ���ͼ
% theta
figure
Q1 = semilogy(theta_interval,theta_fitness,theta_interval,theta_C1,theta_interval,theta_C2,theta_interval,theta_C3,theta_interval,theta_C4);
set(Q1,'linewidth',1.1);

xlabel('��ȡֵ��Χ','fontweight','bold');
ylabel('����(��λ��Ԫ)','fontweight','bold');

legend('�ܳɱ�','����ɱ�','ĩ�����ͳɱ�','������Ӫ�ɱ�','���ʶ���ʧ�ɱ�','Location','best')


grid on
set(gca,'linewidth',1.1)    % ����������������
    
% beta
figure
Q2 = semilogy(beta_interval,beta_fitness,beta_interval,beta_C1,beta_interval,beta_C2,beta_interval,beta_C3,beta_interval,beta_C4);
set(Q2,'linewidth',1.1);

xlabel('��ȡֵ��Χ','fontweight','bold');
ylabel('����(��λ��Ԫ)','fontweight','bold');

legend('�ܳɱ�','����ɱ�','ĩ�����ͳɱ�','������Ӫ�ɱ�','���ʶ���ʧ�ɱ�','Location','best')

grid on
set(gca,'linewidth',1.1)    % ����������������
    


% %% ���Ʋ���ͼ
% D=size(distance,1);%����ά��
% [~,peisong] = fun(zbest,distance,data,theta,beta);
% 
% gyqk = zeros(D);  % ��Ӧ���
% for i = 1: size(peisong,2)
%     gyqk(peisong(1,i),i) = data(i,5);
% end
% 
% % �ҳ�����С����ǹ���С��
% gonghuoxiaoqu = find(zbest == 1);
% feigonghuo = find(zbest == 0);
% 
% % ���ƹ���С��
% figure
% plot(data(gonghuoxiaoqu,1),data(gonghuoxiaoqu,2),'s','markersize',6,'markerfacecolor',[0.5,0.5,0.5],'markeredgecolor','b')
% hold on
% 
% % ��������С��
% plot(data(feigonghuo,1),data(feigonghuo,2),'o','markersize',3,'markerfacecolor','g','markeredgecolor','g')
% hold on
% 
% % ����·��
% for i = 1:size(peisong,2)
%     line([data(peisong(1,i),1),data(i,1)],[data(peisong(1,i),2),data(i,2)],'color','green')
%     hold on
% end
% 
% % ��������
% axis([min(data(:,1))-500,max(data(:,1))+500,min(data(:,2))-500,max(data(:,2))+500])
% xlabel('������(��λ:m)','fontweight','bold')
% ylabel('������(��λ:m)','fontweight','bold')
% set(gca,'linewidth',1.1)