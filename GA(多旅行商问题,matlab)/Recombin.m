%% �������
% ����
%SelCh  ��ѡ��ĸ���
%Pc     �������
%�����
% SelCh �����ĸ���
function SelCh=Recombin(SelCh,Pc)
NSel=size(SelCh,1);   % NSel Ϊ��ѡ��ĸ�������
for i=1:2:NSel-mod(NSel,2)   % �ٽ��������彻��
    if Pc>=rand %�������Pc
        [SelCh(i,:),SelCh(i+1,:)]=intercross(SelCh(i,:),SelCh(i+1,:));
    end
end

%%  ���溯��
function [a,b]=intercross(a,b)
L=length(a);
cross_p=randperm(L-2,1)+1;  %����λ��
% ����Ϊ���沽��
c1=a(1:cross_p);
a(1:cross_p)=b(1:cross_p);
b(1:cross_p)=c1;
c2=a(cross_p+1:length(a));
a(cross_p+1:length(a))=b(cross_p+1:length(b));
b(cross_p+1:length(b))=c2;


