% =========== load data =======

% load('Xraw.mat');

tic

for i = 1:5
    for j = 1:5
        
        t = squeeze( Xraw(:,i,j) )';
        p = polyfit(1:924,t,10);
        t_h = polyval(p,1:924);        
        T(:,i,j) = t_h;
    end
    disp(i)
    toc;
end
save('T.mat','T');

% =============== different position ======
figure;
subplot(1,2,1)
for i = 1:5
    plot(Xraw(:,2,i),'o')
    hold on;
end
title('unsmooth')

subplot(1,2,2)
for i = 1:5
    plot(T(:,2,i),'o')
    hold on;
end
title('smoothed')
xlabel('same anttena, different positon')


% =============== different ant ======
figure;
subplot(1,2,1)
for i = 1:5
    plot(Xraw(:,i,1),'o')
    hold on;
end
title('unsmooth')

subplot(1,2,2)
for i = 1:5
    plot(T(:,i,1),'o')
    hold on;
end
title('smoothed')
xlabel('same position, different anttena')

