% multi poly smooth & downsample by 14

load('Xraw.mat');

tic
for i = 1:56
    for j = 1:4979
        
        t = squeeze( Xraw(:,i,j) )';
        p = polyfit(1:924,t,10);
        t_h = polyval(p,1:924);
        
        X(:,i,j) = t_h(1:14:end);
    end
    disp(i)
    toc;
end

save('X.mat','X')