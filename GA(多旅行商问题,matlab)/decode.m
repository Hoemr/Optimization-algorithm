function [car_path,upload_num,dist_all] = decode(chrom,demand,N,max_vc,max_v,D)
% �ú������ڽ��룬�����chromΪһ������,NΪ������,max_vcΪ�������
% max_vΪÿһ�ֳ�������������,demandΪÿ�����������,DΪ�������
%���Եõ���
% 1�������Ŀ�꺯��ֵ
% 2�������Ӧ���·�����䳵������

%% ��������
% �ֿ�
renwu = chrom(find(chrom<=N));   % �ֳ���������
cheliang = chrom(find(chrom>N));     % �ֳ����ĳ���

j = 1;   % ������¼�ڼ�����
num_cheliang = zeros(1,length(max_v));   % ÿһ�����乤�ߵ�ʹ����
k = 1;   % ������¼��ǰ�ǵڼ���·��
m = 1;   % ������¼��ǰ·���Ѿ��ж��ٸ������������
upload_num = 0;   % ����·�����Ѿ�ת�صĻ�����
dist_all = 0;   % ����·�����ܳ���

%% ѡ���ʼ·����ͨ���߲�ȷ���������ת����
car_path = [];   % �������һ������Ϊ·�����乤�����ͼ�·��(��һ��Ϊ���ͣ�1λ������2Ϊ�𳵣�3λ�ɻ�)
    if cheliang(j)>N && cheliang(j) <= N+max_v(1)
        num_cheliang(1) = num_cheliang(1)+1;   % ʹ�õ�һ�ֳ�������Ŀ����1
        capacity = max_vc(1);    % ��һ����·���������
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
    
%% ��������
for i = 1:length(renwu)
    if upload_num(k)+demand(renwu(i)+1,2) <= capacity   % ������װ��ǰ���������Ļ�
        car_path(k,m+1) = renwu(i);   % ���õ�k��·����m�����������
        if m == 1
            dist_all(k) = D(1,renwu(i)+1);  % �����㵽��һ�������ľ���
        else
            dist_all(k) = dist_all(k)+D(renwu(i-1)+1,renwu(i)+1);  % ǰһ�����󵽵���ǰ�����ľ���
        end
        m = m+1;     % �����õ���һ�������
        upload_num(k) = upload_num(k)+demand(renwu(i)+1,2);  % ��ǰ·���ϵĻ���ת����
    else
        % ��Ҫ�������乤����
        j = j+1;
        % ��Ҫ��һ��·��
        k = k+1;
        % ����·���Ѿ������������������Ϊ1
        m = 1;
        % װ������0
        upload_num(k) = 0;
        % ȷ�����������乤��
        if cheliang(j)>N && cheliang(j) <= N+max_v(1)
        num_cheliang(1) = num_cheliang(1)+1;   % ʹ�õ�һ�ֳ�������Ŀ����1
        capacity = max_vc(1);    % ��һ����·���������
        car_path(k,1) = 1;  % ����·�������乤������
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

