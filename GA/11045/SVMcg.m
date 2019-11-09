function [bestacc,bestc,bestg] = SVMcg(train_label,train,cmin,cmax,gmin,gmax,v,cstep,gstep,accstep)
% ʹ��˵��.����:
% [bestacc,bestc,bestg] = SVMcg(train_label,train,cmin,cmax,gmin,gmax,v,cstep,gstep,accstep)
%
% train_label:ѵ�� ����ǩ.Ҫ����libsvm��������Ҫ��һ��.
% train:ѵ����.Ҫ����libsvm��������Ҫ��һ��.
% cmin:�ͷ�����c�ı仯��Χ����Сֵ(ȡ��2Ϊ�׵Ķ�����),�� c_min = 2^(cmin).Ĭ��Ϊ -5
% cmax:�ͷ�����c�ı仯��Χ�����ֵ(ȡ��2Ϊ�׵Ķ�����),�� c_max = 2^(cmax).Ĭ��Ϊ 5
% gmin:����g�ı仯��Χ����Сֵ(ȡ��2Ϊ�׵Ķ�����),�� g_min = 2^(gmin).Ĭ��Ϊ -5
% gmax:����g�ı仯��Χ����Сֵ(ȡ��2Ϊ�׵Ķ�����),�� g_min = 2^(gmax).Ĭ��Ϊ 5
%
% v:cross validation�Ĳ���,�������Լ���Ϊ�����ֽ���cross validation.Ĭ��Ϊ 3
% cstep:����c�����Ĵ�С.Ĭ��Ϊ 1
% gstep:����g�����Ĵ�С.Ĭ��Ϊ 1
% accstep:�����ʾ׼ȷ��ͼʱ�Ĳ�����С. Ĭ��Ϊ 1.5

%% about the parameters of SVMcg
if nargin < 10
    accstep = 1.5;
end
if nargin < 8
    accstep = 1.5;
    cstep = 1;
    gstep = 1;
end
if nargin < 7
    accstep = 1.5;
    v = 3;
    cstep = 1;
    gstep = 1;
end
if nargin < 6
    accstep = 1.5;
    v = 3;
    cstep = 1;
    gstep = 1;
    gmax = 5;
end
if nargin < 5
    accstep = 1.5;
    v = 3;
    cstep = 1;
    gstep = 1;
    gmax = 5;
    gmin = -5;
end
if nargin < 4
    accstep = 1.5;
    v = 3;
    cstep = 1;
    gstep = 1;
    gmax = 5;
    gmin = -5;
    cmax = 5;
end
if nargin < 3
    accstep = 1.5;
    v = 3;
    cstep = 1;
    gstep = 1;
    gmax = 5;
    gmin = -5;
    cmax = 5;
    cmin = -5;
end
%% X:c Y:g cg:acc
[X,Y] = meshgrid(cmin:cstep:cmax,gmin:gstep:gmax);
[m,n] = size(X);
cg = zeros(m,n);
%% record acc with different c & g,and find the bestacc with the smallest c
bestc = 0;
bestg = 0;
bestacc = 0;
basenum = 2;
for i = 1:m
    for j = 1:n
        cmd = ['-v ',num2str(v),' -c ',num2str( basenum^X(i,j) ),' -g ',num2str( basenum^Y(i,j) ),' -s 3 -p 0.01 -t 0'];
        cg(i,j) = libsvmtrain(train_label, train, cmd);

        if cg(i,j) > bestacc
            bestacc = cg(i,j);
            bestc = basenum^X(i,j);
            bestg = basenum^Y(i,j);
        end
        if ( cg(i,j) == bestacc && bestc > basenum^X(i,j) )
            bestacc = cg(i,j);
            bestc = basenum^X(i,j);
            bestg = basenum^Y(i,j);
        end

    end
end
%% to draw the acc with different c & g
figure
surf(X,Y,cg)
xlabel('����c����2�Ķ���log2c','fontweight','bold')
ylabel('����g����2�Ķ���log2g','fontweight','bold')
zlabel('���������','fontweight','bold')
set(gca,'linewidth',1.1)
grid on
box on