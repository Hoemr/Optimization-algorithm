%% ��ʼ����Ⱥ
%���룺
% NIND����Ⱥ��С
% N��   ����Ⱦɫ�峤�ȣ�����Ϊ���еĸ�����  
%�����
%��ʼ��Ⱥ
function chrom=initpop(NIND,N)
chrom=zeros(NIND,N);%���ڴ洢��Ⱥ
for i=1:NIND
     chrom(i,:)=randperm(N);%������ɳ�ʼ��Ⱥ
end
