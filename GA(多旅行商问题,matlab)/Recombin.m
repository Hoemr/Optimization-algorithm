%% 交叉操作
% 输入
%SelCh  被选择的个体
%Pc     交叉概率
%输出：
% SelCh 交叉后的个体
function SelCh=Recombin(SelCh,Pc)
NSel=size(SelCh,1);   % NSel 为被选择的个体数量
for i=1:2:NSel-mod(NSel,2)   % 临近两个个体交叉
    if Pc>=rand %交叉概率Pc
        [SelCh(i,:),SelCh(i+1,:)]=intercross(SelCh(i,:),SelCh(i+1,:));
    end
end

%%  交叉函数
function [a,b]=intercross(a,b)
L=length(a);
cross_p=randperm(L-2,1)+1;  %交叉位置
% 以下为交叉步骤
c1=a(1:cross_p);
a(1:cross_p)=b(1:cross_p);
b(1:cross_p)=c1;
c2=a(cross_p+1:length(a));
a(cross_p+1:length(a))=b(cross_p+1:length(b));
b(cross_p+1:length(b))=c2;


