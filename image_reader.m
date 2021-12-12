clear;
clc;

enteryNumber = 162;

data = readtable('train.csv');
data = data{:, :};
img = 255 - imread("number ttest.png");

temp = zeros(28);
temp2 = zeros(28);

for i = 1:28
    for j = 1:28
        temp(j,i) = data(enteryNumber, i+(j-1)*28 );
    end
end

for i = 1:28
    for j = 1:28
        temp2(i,j) = mean(img(i,j));
    end
end

index = 1;
clf;
hold on;

for i = 1:28
    for j = 1:28
        c = (255-temp(i,j))/255;
        color = [c, c, c];
        plot(j, 29-i, "s", "Color", color, 'MarkerSize',10,'MarkerFaceColor',color)
        index = index + 1;
    end
end

axis([0 29 0 29])

