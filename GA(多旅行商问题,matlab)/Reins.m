 %% �ز����Ӵ�������Ⱥ
 %���룺
 %Chrom  ��������Ⱥ
 %SelCh  �Ӵ���Ⱥ
 %ObjV   ������Ӧ��
 %���
 % Chrom  ��ϸ������Ӵ���õ�������Ⱥ
function chrom=Reins(Chrom,SelCh,ObjV)
NIND=size(Chrom,1);  % ԭʼ��Ⱥ��С
NSel=size(SelCh,1);   % ��ѡ�����Ⱥ��С
[~,index]=sort(ObjV);
% ����ʼ��Ⱥ�����������NIND-NSel�������������Ⱥ���ټ���ѡ��õ��ĸ���ϲ���һ����Ⱥ
chrom=[Chrom(index(1:NIND-NSel),:);SelCh];

