% =========== load data =======
load('/home/jianyuan/Codes/1_CTW2/Figs/5_DNN/X.mat')
load('/home/jianyuan/Codes/1_CTW2/Figs/5_DNN/Y.mat')





% =========== find neibor =======
y1 = Y(1,:);
idx = []
for i = 2:4979
    if sqrt( sum(( Y(i,:) - Y(1,:)).^2) ) < 3
        idx = [idx,i]
    end
end

% =========== plot =======
figure
scatter(Y(:,1),Y(:,2))
hold on;
scatter(Y(idx,1),Y(idx,2))
hold on;
scatter(Y(1,1),Y(1,2),'s')

% =========== plot CSI =======
figure;
for i = 1:7
    plot(  squeeze( X(idx(i),5,:)) ,'o'  )
    hold on;
end



        