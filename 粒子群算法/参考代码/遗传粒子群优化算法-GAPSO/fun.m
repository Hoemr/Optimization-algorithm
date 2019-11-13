function y = fun(x)
%函数用于计算粒子适应度值
%x           input           输入粒子 
%y           output          粒子适应度值 
[row,col] = size(x);
sum1=0;
sum2=1;
sum3=0;

%Sphere Model函数
%for i=1:col
  % sum1=sum1+x(i)^2;
%end
%y=sum1;


%Schwefel's Problem 2.22函数
%for i=1:col
    %sum1=sum1+norm(x(i));
    %sum2=sum2*norm(x(i));
%end
%y=sum1+sum2;



%Generalized Rosenbrock函数
%for i=1:col-1
  %sum1=sum1+100*(x(i+1)-x(i).^2).^2+(x(i)-1).^2;
%end
%y=sum1;



%Generalized Rastrigin函数
for i=1:col
   sum1=sum1+x(i)^2-10*cos(2*pi*x(i))+10;
end
y=sum1;



%Ackley函数
%for i=1:col
    %sum1=sum1+x(i)^2;
    %sum3=sum3+cos(2*pi*x(i));
%end
%y=-20*exp(-0.2*sqrt(sum1./30))-exp(sum3./30)+20+exp(1);
    
