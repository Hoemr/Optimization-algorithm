function [car_path,upload_num,dist_all] = decode(chrom,demand,N,max_vc,max_v,D)
% 该函数用于解码，传入的chrom为一个个体,N为结点个数,max_vc为最大载重
% max_v为每一种车辆的最大可用量,demand为每个点的需求量,D为距离矩阵
%可以得到：
% 1：个体的目标函数值
% 2：个体对应解的路径与其车辆安排

%% 参数设置
% 分开
renwu = chrom(find(chrom<=N));   % 分出来的任务
cheliang = chrom(find(chrom>N));     % 分出来的车辆

j = 1;   % 用来记录第几辆车
num_cheliang = zeros(1,length(max_v));   % 每一种运输工具的使用量
k = 1;   % 用来记录当前是第几条路径
m = 1;   % 用来记录当前路径已经有多少个货物需求点了
upload_num = 0;   % 所有路径上已经转载的货物量
dist_all = 0;   % 所有路径的总长度

%% 选择初始路径交通工具并确定下来最大转载量
car_path = [];   % 这里面第一个数字为路径运输工具类型及路径(第一个为类型，1位货车，2为火车，3位飞机)
    if cheliang(j)>N && cheliang(j) <= N+max_v(1)
        num_cheliang(1) = num_cheliang(1)+1;   % 使用第一种车辆的数目增加1
        capacity = max_vc(1);    % 这一条线路最大运输量
        car_path(k,1) = 1;
    else if cheliang(j) >N+max_v(1) && cheliang(j) <= N+max_v(1)+max_v(2)
            num_cheliang(2) = num_cheliang(2)+1;
            capacity = max_vc(2);
            car_path(k,1) = 2;
        else if cheliang(j) > N+max_v(1)+max_v(2) && cheliang(j) <= N+max_v(1)+max_v(2)+max_v(3)
                num_cheliang(3) = num_cheliang(3)+1;
                capacity = max_vc(3);
                car_path(k,1) = 3;
            end
        end
    end
    
%% 遍历任务
for i = 1:length(renwu)
    if upload_num(k)+demand(renwu(i)+1,2) <= capacity   % 可以再装当前货物需求点的货
        car_path(k,m+1) = renwu(i);   % 设置第k条路径第m个货物需求点
        if m == 1
            dist_all(k) = D(1,renwu(i)+1);  % 供货点到第一个需求点的距离
        else
            dist_all(k) = dist_all(k)+D(renwu(i-1)+1,renwu(i)+1);  % 前一个需求到到当前需求点的距离
        end
        m = m+1;     % 更换得到下一个需求点
        upload_num(k) = upload_num(k)+demand(renwu(i)+1,2);  % 当前路径上的货物转载量
    else
        % 需要更换运输工具了
        j = j+1;
        % 需要换一个路径
        k = k+1;
        % 这条路径已经供货的需求点数量清为1
        m = 1;
        % 装载量清0
        upload_num(k) = 0;
        % 确认是哪种运输工具
        if cheliang(j)>N && cheliang(j) <= N+max_v(1)
        num_cheliang(1) = num_cheliang(1)+1;   % 使用第一种车辆的数目增加1
        capacity = max_vc(1);    % 这一条线路最大运输量
        car_path(k,1) = 1;  % 这条路径的运输工具类型
        else if cheliang(j) >N+max_v(1) && cheliang(j) <= N+max_v(1)+max_v(2)
            num_cheliang(2) = num_cheliang(2)+1;
            capacity = max_vc(2);
            car_path(k,1) = 2;
        else if cheliang(j) > N+max_v(1)+max_v(2) && cheliang(j) <= N+max_v(1)+max_v(2)+max_v(3)
                num_cheliang(3) = num_cheliang(3)+1;
                capacity = max_vc(3);
                car_path(k,1) = 3;
            end
        end
        end
    end
end

