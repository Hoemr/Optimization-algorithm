% ����:
%FitnV  �������Ӧ��ֵ
%Nsel   ��ѡ��������Ŀ
% ���:
%NewChrIx  ��ѡ������������
function NewChrIx = Sus(FitnV,Nsel)
[Nind,~] = size(FitnV);   % NindΪ����������Ҳ���������ֵ
cumfit = cumsum(FitnV);   % �ۼ������Ӧ��
trials = cumfit(Nind) / Nsel * (rand + (0:Nsel-1)');
Mf = cumfit(:, ones(1, Nsel));
Mt = trials(:, ones(1, Nind))';
[NewChrIx, ~] = find(Mt < Mf & [ zeros(1, Nsel); Mf(1:Nind-1, :) ] <= Mt);
[~, shuf] = sort(rand(Nsel, 1));
NewChrIx = NewChrIx(shuf);



