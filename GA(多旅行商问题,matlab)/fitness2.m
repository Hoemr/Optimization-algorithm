function [all_fitness,best_fitness,all_obj,best_obj,best_chrom,best_car_path] = fitness2(chrom,demand,N,max_vc,max_v,cb,cv,D)
% chrom为整个种群,N为任务需求点个数,max_vc为最大载重
% max_v为每一种交通工具的最大可用量,demand为每个点的需求量
% cb为固定出车成本,cv为单位运输成本

%% 参数设置
num  = size(chrom,1);   % 个体数
best_fitness = 0;% 种群最佳适应度
best_obj = 0;
best_chrom = [];   % 最优解对应的个体
best_car_path = [];  % 最优解对应的路径以及交通工具安排情况
all_fitness = zeros(1,num);
all_obj = zeros(1,num);
%% 正式计算适应度
for i = 1:num
    % 解码得到路径及交通工具安排情况
    [car_path,upload_num,dist_all] = decode(chrom(i,:),demand,N,max_vc,max_v,D);
    % 计算个体对应的目标函数值
    obj = 0;
    % 遍历路径计算总费用
    for k = 1:size(car_path,1)   % 遍历每一条路径
        % 加上固定出车成本费,car_path(k,1)为第k条路径的运输工具序号
        obj = obj+cb(car_path(k,1));
        % 加上这一条路径的运输费用
        obj = obj + cv(car_path(k,1))*upload_num(k)*dist_all(k);
    end
    % 计算适应度
    fit = 1/obj;
    all_fitness(i) = fit;
    all_obj(i) = obj;
    if fit > best_fitness   % 优于目前最优个体则更新最优个体
        best_fitness = fit;
        best_obj = obj;
        best_chrom = chrom(i,:);
        best_car_path = car_path;
    end
end
end

