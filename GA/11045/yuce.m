clear,clc
%% ����Ԥ����
% ��ȡ����
x_train = xlsread('train_data.xlsx','Sheet1','A2:C21');
x_pred = xlsread('predict_data.xlsx','Sheet1','H3:J22');
% ���ݹ�һ��,ӳ��x->y����һ���Ľ��Ϊy(����һ)
[y_train(:,1),ps(1)] = mapminmax(x_train(:,1)');   % ��ѡƷλ��һ��
[y_train(:,2),ps(2)] = mapminmax(x_train(:,2)');    % ����Ʒλ��һ��
[y_train(:,3),ps(3)] = mapminmax(x_train(:,3)');    % ѡ��ȹ�һ��

%ѡ��ѵ����������
num_train = floor(0.7*size(y_train(:,1)));
%�������ѡ������
choose = randperm(length(y_train(:,1)));   
train_data = y_train(choose(1:num_train),:);   % ���е�ѵ����
% gscatter(train_data(:,1),train_data(:,2),train_data(:,3));
output_train = train_data(:,end);   % ѵ���������
input_train = train_data(:,1:2);

test_data = y_train(choose(num_train+1:end),:);   % ���еĲ��Լ�
output_test = test_data(:,end);    % ���Լ��������ʵֵ
input_test = test_data(:,1:2);    % ���Լ�������

%% ���ý�����֤ѡ��������������
[bestacc,bestc,bestg] = SVMcg(output_train,input_train,-8,8,-8,8,3,0.2,0.2);

%% ʹ��ѡȡ�Ĳ���ѵ��ģ�ͣ���������
% ѵ��ģ��
options = ['-c ',num2str(bestc),' -g ',num2str(bestg),' -s 3 -p 0.01 -n 0.1 -t 0'];
model = libsvmtrain(output_train,input_train,options);

% ---------------------���Լ�------------------------%
% �����Լ���Ԥ��û�
[py_test,~,~] = libsvmpredict(output_test,input_test,model);
% ���Լ�����ķ���һ��
py_test_guiyi = mapminmax('reverse',py_test,ps(3));  % Ԥ�����Ĺ�һ��
output_test_guiyi = mapminmax('reverse',output_test,ps(3));  % ���Լ���ʵ����Ĺ�һ��

% ------------------��ҪԤ�������------------------%
% ����ҪԤ�����������һ��
y_pred(:,1) = mapminmax('apply',x_pred(:,1)',ps(1));   % ��ѡƷλ��һ��
y_pred(:,2) = mapminmax('apply',x_pred(:,2)',ps(2));    % ����Ʒλ��һ��
y_pred(:,3) = mapminmax('apply',x_pred(:,3)',ps(3));    % ѡ��ȹ�һ��

% Ԥ������
[py_pred,~,dec_value] = libsvmpredict(y_pred(:,3),y_pred(:,1:2),model);
% ��Ԥ����������һ��
py_pred_guiyi = mapminmax('reverse',py_pred,ps(3));

% Ԥ������ͼ
figure;
plot(1:length(y_pred(:,3)),x_pred(:,3),'r-*','linewidth',1.1);  % ��ʵֵ
set(gca,'linewidth',1.4)
hold on;
plot(1:length(y_pred(:,3)),py_pred_guiyi,'m-.o','linewidth',1.1);   % Ԥ��ֵ

legend('��ʵֵ','����ʽ�˺���Ԥ��ֵ');
grid on;
title('����ʽ�˺���Ԥ�����','fontweight','bold')
xlabel('���Լ��������','fontweight','bold')
ylabel('ѡ���','fontweight','bold')
grid on

%% ����Ϊ�������������ϸ˵��
%% libsvm���ø�ʽ
% model = libsvmtrain(train_label,train_data,options);
% [predict_label,accuracy/mse,dec_value] = libsvmpredict(test_label,test_data,model);

%% options��������
% - s svm���ͣ�SVMģ���������ͣ�Ĭ��ֵΪ0��
%     0��C - SVC
%     1:nu - SVC
%     2��one - class SVM
%     3: epsilon - SVR    ����Ǵ��гͷ����SVRģ��
%     4: nu - SVR
% - t �˺������ͣ��˺����������ͣ�Ĭ��ֵΪ2��
%     0:���Ժ˺��� u'v
%     1:����ʽ�˺���(r *u'v + coef0)^degree
%     2:RBF �˺��� exp( -r|u - v|^2)
%     3:sigmiod�˺��� tanh(r * u'v + coef0)
% - d degree:�˺����е� degree �������ã���Զ���ʽ�˺�����Ĭ��ֵΪ3��
% - g r(gama):gΪ�˺������˺����е�gama�������ã���Զ���ʽ/sigmoid �˺���/RBF/��Ĭ��ֵΪ������Ŀ�ĵ�����
% - r coef0:�˺����е�coef0��������(��Զ���ʽ/sigmoid�˺�����Ĭ��ֵΪ0)
% - c cost(�ͷ�����):���� C - SVC,epsilon - SVR �� nu - SVR�Ĳ�����Ĭ��ֵΪ1��
% - n nu:���� nu-SVC ��one - class SVM �� nu - SVR�Ĳ���
% - p epsilon:���� epsilon - SVR ����ʧ������ֵ��Ĭ��ֵΪ0.1��
% - m cachesize:���� cache �ڴ��С���� MB Ϊ��λ��Ĭ��ֵΪ100��
% - e eps:�����������ֹ��ֵ��Ĭ��ֵΪ0.001��
% - h shrinking���Ƿ�ʹ������ʽ��0��1��Ĭ��ֵΪ1��
% - wi weight:���õڼ���Ĳ��� C Ϊ weight * C������ C - SVC �е� C��Ĭ��ֵΪ1��
% - v n:n - fold ��������ģʽ��nΪ������������ڵ���2

%% svmtrain�����model
% optimization finished, #iter = 505	%iterΪ��������
% nu = 0.679187						%nu ����ѡ��ĺ˺������͵Ĳ���
% obj = -113.509601, rho = 0.051363   %objΪSVM�ļ�ת��Ϊ�Ķ��ι滮���õ�����Сֵ
% nSV = 259, nBSV = 18				%nSV Ϊ��׼֧����������(0<a[i]<c)��nBSVΪ�߽��ϵ�֧����������(a[i]=c)��
% Total nSV = 259						%nBSVΪ�߽��ϵ�֧����������(a[i]=c)������������˵����Ϊֻ��һ������ģ��Total nSV = nSV��
%                                     %���Ƕ��ڶ��࣬����Ǹ�������ģ�͵�nSV֮��
% model = 
%   ���������ֶε� struct:			
% 
%     Parameters: [5��1 double]		%�ṹ����������α������ -s -t -d -g -r�Ȳ���
%       nr_class: 2				    %����ĸ���   
%        totalSV: 259				    %�ܵ�֧����������
%            rho: 0.0514				%b=-model.rho
%          Label: [2��1 double]
%     sv_indices: [259��1 double]
%          ProbA: []
%          ProbB: []
%            nSV: [2��1 double]	    %ÿһ���֧�������ĸ���
%        sv_coef: [259��1 double]	    %֧��������ϵ��
%            SVs: [259��13 double]	    %�����֧����������ϡ��������ʽ�洢
% ���У�
%    w*x+b=0 ��
%      w=model.SVs'*model.sv_coef
%      b=-model.rho
% w�Ǹ�ά�ռ��з��� ��ƽ��ķ�������b�ǳ����

%% model.Parameters������
% Parameters =
%          0
%     2.0000
%     3.0000
%     2.8000
%          0
% ```c
% -  model.Paraments ����������ϵ�������Ϊ
% 
% ```c
% -s svm���ͣ�SVMģ���������ͣ�Ĭ��ֵΪ0��
%  
% - t �˺������ͣ��˺����������ͣ�Ĭ��ֵΪ2��
%  
% - d degree:�˺����е� degree �������ã���Զ���ʽ�˺�����Ĭ��ֵΪ3��
%  
% - g r(gama):�˺����е�gama�������ã���Զ���ʽ/sigmoid �˺���/RBF/��Ĭ��ֵΪ������Ŀ�ĵ�����
%  
% - r coef0:�˺����е�coef0��������(��Զ���ʽ/sigmoid�˺�����Ĭ��ֵΪ0)

%% model.Label model.nr_class
% model.label ��ʾ���ݼ������ı�ǩ����ʲô
% model.nr_class ��ʾ���ݼ�ְ���ж��ٸ����

%% model.totalSV model.nSV
% model.total SV �����ܹ���֧������������Ŀ������һ��259����
% model.nSV ����ÿ��������֧����������Ŀ��model.nSV �������˳���Ǻ� model.label ���Ӧ�� ��ǩΪ1������118������ǩΪ-1������ 141 ����

%% model.sv_coef model.SVs model.rho
% sv_coef: [259x1 double]
%     SVs: [259x13 double]<br>  model.rho = 0.0514
% 
% sv_coef����װ����259��֧�������ھ��ߺ����е�ϵ����
% model.SVs ��װ����259��֧��������
% model.rho = 0.0514   �Ǿ��ߺ����еĳ�������෴����

%% accuracy
% ���ص�accuracy���ϵ��µ����������ǣ�
% 1������׼ȷ�ʣ������������õ��Ĳ���ָ�ꣻ
% 2��ƽ��ƽ���� mean squared error,MSE�����ع��������õ��Ĳ���ָ�ꣻ
% 3��ƽ�����ϵ���� squared correlation coefficient ��R^2�����ع��������õ��Ĳ���ָ�ꡣ