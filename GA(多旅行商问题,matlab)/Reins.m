 %% 重插入子代的新种群
 %输入：
 %Chrom  父代的种群
 %SelCh  子代种群
 %ObjV   父代适应度
 %输出
 % Chrom  组合父代与子代后得到的新种群
function chrom=Reins(Chrom,SelCh,ObjV)
NIND=size(Chrom,1);  % 原始种群大小
NSel=size(SelCh,1);   % 被选择的种群大小
[~,index]=sort(ObjV);
% 将初始种群里面最优秀的NIND-NSel个个体放入新种群，再加上选择得到的个体合并成一个种群
chrom=[Chrom(index(1:NIND-NSel),:);SelCh];

