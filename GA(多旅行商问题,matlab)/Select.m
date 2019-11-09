%% 选择操作
%输入
%Chrom 种群
%FitnV 适应度值
%GGAP：代沟
%输出
%SelCh  被选择的个体
function SelCh=Select(Chrom,FitnV,GGAP)
NIND=size(Chrom,1);   % 种群大小
NSel=max(floor(NIND*GGAP+.5),2);   % 被选择的个体数目
ChrIx=Sus(FitnV,NSel);   % ChrIx为被选择个体的索引号
SelCh=Chrom(ChrIx,:);   % 根据索引选择个体
