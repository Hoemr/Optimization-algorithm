function [all_fitness,best_fitness,all_obj,best_obj,best_chrom,best_car_path] = fitness2(chrom,demand,N,max_vc,max_v,cb,cv,D)
% chromΪ������Ⱥ,NΪ������������,max_vcΪ�������
% max_vΪÿһ�ֽ�ͨ���ߵ���������,demandΪÿ�����������
% cbΪ�̶������ɱ�,cvΪ��λ����ɱ�

%% ��������
num  = size(chrom,1);   % ������
best_fitness = 0;% ��Ⱥ�����Ӧ��
best_obj = 0;
best_chrom = [];   % ���Ž��Ӧ�ĸ���
best_car_path = [];  % ���Ž��Ӧ��·���Լ���ͨ���߰������
all_fitness = zeros(1,num);
all_obj = zeros(1,num);
%% ��ʽ������Ӧ��
for i = 1:num
    % ����õ�·������ͨ���߰������
    [car_path,upload_num,dist_all] = decode(chrom(i,:),demand,N,max_vc,max_v,D);
    % ��������Ӧ��Ŀ�꺯��ֵ
    obj = 0;
    % ����·�������ܷ���
    for k = 1:size(car_path,1)   % ����ÿһ��·��
        % ���Ϲ̶������ɱ���,car_path(k,1)Ϊ��k��·�������乤�����
        obj = obj+cb(car_path(k,1));
        % ������һ��·�����������
        obj = obj + cv(car_path(k,1))*upload_num(k)*dist_all(k);
    end
    % ������Ӧ��
    fit = 1/obj;
    all_fitness(i) = fit;
    all_obj(i) = obj;
    if fit > best_fitness   % ����Ŀǰ���Ÿ�����������Ÿ���
        best_fitness = fit;
        best_obj = obj;
        best_chrom = chrom(i,:);
        best_car_path = car_path;
    end
end
end

