figure;

for i = 1:9
    
    %     h5disp( strcat('./file_',num2str(i),'.hdf5'))
    x = h5read(strcat('/home/jianyuan/Codes/1_CTW2/label/file_',...
        num2str(i),'.hdf5'),'/Pos');
    
    subplot(3,3,i)
    scatter(x(1,:), x(2,:))
end

