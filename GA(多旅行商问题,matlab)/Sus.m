% 输入:
%FitnV  个体的适应度值
%Nsel   被选择个体的数目
% 输出:
%NewChrIx  被选择个体的索引号
function NewChrIx = Sus(FitnV,Nsel)
[Nind,~] = size(FitnV);   % Nind为个体总数，也即索引最大值
cumfit = cumsum(FitnV);   % 累计求和适应度
trials = cumfit(Nind) / Nsel * (rand + (0:Nsel-1)');
Mf = cumfit(:, ones(1, Nsel));
Mt = trials(:, ones(1, Nind))';
[NewChrIx, ~] = find(Mt < Mf & [ zeros(1, Nsel); Mf(1:Nind-1, :) ] <= Mt);
[~, shuf] = sort(rand(Nsel, 1));
NewChrIx = NewChrIx(shuf);



