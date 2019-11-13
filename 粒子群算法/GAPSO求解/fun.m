function [fitness,peisong] = fun(pop,distance,data,theta,beta)
%�������ڼ���������Ӧ��ֵ
%pop           input           �������� 
%fitness       output          ������Ӧ��ֵ 

r = 3000;   % ����뾶
rho = 1.2;    % ��������ϵ��
F = 187500;    % �����꽨��ɱ�
c = 0.007;    % ��λ���뵥λ���������ͷ���
P = 65;    % ��λ���ʵ��̲�Ʒ��ƽ���۸�
M = 50000;    % ������Ľ�������  
v = 8.33;   % �ٶ�    ��Ҫ��ʾ���ʶ���ʱ��ı仯����, T = d/v
lambda = 4;   % �����������Ӫ�ɱ�����Ӫ�������ϵ��

g = 1;    % ���¶ȱ仯���仯�ķ�Ӧ����

peisong = zeros(2,length(pop));    % ��һ��Ϊÿ��С��������С�����ڶ���Ϊÿ����ӦС��Ŀǰ�Ѿ���Ӧ����
peisong(2,find(pop == 1)) = data(find(pop == 1),5)';   

%% ----------------ȷ��ÿ��С��������С�� -------------------%%
for i = 1:length(pop)   % ����ÿһ��С��
    if pop(i)==1   % ����ǹ�Ӧ��
        peisong(1,i) = i;
    else
        % ���i���ǹ�Ӧ�㣬��i��j�ľ����������
        [~,index] = sort(distance(i,:));
        for j = 2:length(index)   % ��һ�����϶���i�����վ���ѡ������С��
            if pop(index(j)) == 1  % һ��Ҫ�����͵�
                % ���û�г�������
                if peisong(2,index(j))+ data(i,5) <= M && distance(i,index(j)) <= r
                    peisong(1,i) = index(j);
                    peisong(2,index(j)) = peisong(2,index(j)) + data(i,5);
                    break
                end
            end
        end
    end
end

if length(find(peisong(1,:) == 0)) >= 1
    fitness = inf;
    return
end
%% --------------------�����ܷ���--------------------------%%
%%%%%%%%%%%%%%% ����ɱ�
C1 = sum(pop) * F;   % ��ӦС����������F

%%%%%%%%%%%%%%% ĩ�����ͳɱ�
C2 = 0;
for j = 1:length(pop)   % ����ÿһ���������,i��ʵҲ�������Ϊ��i������С����Ҳ���յ�
    C2 = C2 + c*distance(j,peisong(1,j))*data(j,5);
end
    
%%%%%%%%%%%%%%% ������Ӫ�ɱ�
C3 = 0;
for i = 1:length(pop)
    if pop(i) == 1
        C3 = C3 + lambda*P*M;
    end
end

%%%%%%%%%%%%%%% ���ʶ���ʧ�ɱ�
C4 = 0;
for j = 1:length(pop)   % ����ÿһ���������
    C4 = C4 + P*data(j,5)*beta*(1 - g*exp(-theta*distance(j,peisong(1,j))/v));
end

%%%%%%%%%%%%%%% �ܳɱ�
C = C1+C2+C3+C4;

fitness = C;

