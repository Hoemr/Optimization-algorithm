%% ѡ�����
%����
%Chrom ��Ⱥ
%FitnV ��Ӧ��ֵ
%GGAP������
%���
%SelCh  ��ѡ��ĸ���
function SelCh=Select(Chrom,FitnV,GGAP)
NIND=size(Chrom,1);   % ��Ⱥ��С
NSel=max(floor(NIND*GGAP+.5),2);   % ��ѡ��ĸ�����Ŀ
ChrIx=Sus(FitnV,NSel);   % ChrIxΪ��ѡ������������
SelCh=Chrom(ChrIx,:);   % ��������ѡ�����
