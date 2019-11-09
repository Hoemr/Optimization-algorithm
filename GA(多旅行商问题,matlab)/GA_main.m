%�Ŵ��㷨VRP�������
%%            ����˵���ͳ�������
%% �������ݣ�
% D���ڵ�������
% NIND�� ��Ⱥ��ģ
% X���ڵ�����
% MAXGEN������������
% max_vc:���������
% Demand:�ڵ��������
% max_d:��������������
% v:�����ٶ�
% cv:�������䵥��
% max_v:�ɹ�������������
%% �Ż�Ŀ�꣺����ɱ���С ������������
%% Լ�����������㳵���������ƣ����㳵������������ƣ�ͬһ�ڵ�ֻ����һ��������,
%%          �����ڵ�λ�õ�����С�ڳ������������               
% clear
% clc
%%               ��ʼ��
% step1 ���س����ͷ�����Ϣ����
load zuobiao_X   %����ڵ�����:���ڵ��ţ�X,Y��������ʼ�ڵ㣩
% ������������
for i = 1:size(X,1)
    for j = 1:size(X,1)
        D(i,j) = sqrt((X(i,2)-X(j,2))^2+(X(i,3)-X(j,3))^2);
    end
end
N=size(X,1)-1;   %���ͽڵ����
gj_num = 3;   % ���乤����������
load demand      %����������󣺡��ڵ��ţ���������������ʼ�ڵ㣩
max_vc = [50,27,13];  %���乤���������
cv = [30,20,10];   %��λ����ɱ�
cb = [200,150,100];   % �̶������ɱ�
%max_d=40;  %������ʻ������
max_v=[6,3,2];   %����ɵ��䳵���������ֵ
best_obj_list = [];

% step2 GA������ʼ��
NIND=150;  %��Ⱥ��С
MAXGEN=20;   %����������
Pc=0.9;   %�������
Pm=0.05;   %�������
GGAP=0.9;  %����
%%               ��ʼ����
% step1 ���ʻ���Ⱥ
chrom=initpop(NIND,N+gj_num);
gen=0;
% ��������ͼ
figure
hold on
box on;
xlim([0,MAXGEN])% x������������ x limit
title('�Ż�����')
xlabel('����')
ylabel('����ֵ')

% �����һ���������Ӧ�ȣ����ҳ����Ÿ��弰��Ӧ�Ľ���Ⱦɫ��
[all_fitness,best_fitness,all_obj,best_obj,best_chrom,best_car_path] = fitness2(chrom,demand,N,max_vc,max_v,cb,cv,D);
best_obj_list = [best_obj_list best_obj];
pre_obj = best_obj;   % ��һ��������Ŀ�꺯��ֵ
for  gen = 1:MAXGEN
    %% ѡ��
    FitnV = all_fitness;
    SelCh=Select(chrom,FitnV,GGAP);
    %% �������
    SelCh=Recombin(SelCh,Pc);
    %% ����
    SelCh=Mutate(SelCh,Pm);
    %% �ز����Ӵ�������Ⱥ
    ObjV = all_obj;
    chrom=Reins(chrom,SelCh,ObjV);  % �ϲ�������Ⱥ
    %% ��������Ⱥ����Ӧ�ȵ�
    [all_fitness,best_fitness,all_obj,best_obj,best_chrom,best_car_path] = fitness2(chrom,demand,N,max_vc,max_v,cb,cv,D);
    best_obj_list = [best_obj_list best_obj];
    %% ��ͼ
    line([gen-1,gen],[pre_obj,best_obj]);
    pause(0.0001)
    pre_obj = best_obj;   % ��һ��������Ŀ�꺯��ֵ
end
%% �������
% ����Ŀ�꺯��ֵ
disp(['����Ŀ�꺯��ֵ:',num2str(best_obj_list(end))])
% ����ʱ���õĽ�ͨ������������·��������
car_num = size(best_car_path,1);
disp(['����ʱ�ܵĽ�ͨ����ʹ����Ϊ��',num2str(car_num)])
