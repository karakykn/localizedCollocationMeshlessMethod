v11 = table2array(results_interfaceInner);
v22 = table2array(results_interfaceOuter);

x1 = v11(:,2); y1 = v11(:,3);
x2 = v22(:,2); y2 = v22(:,3);

tri1 = delaunay(x1,y1);
tri2 = delaunay(x2,y2);

delos = [];
for i=1:size(tri2,1)
    flag = 0;
    for j=1:size(tri2,2);
        if tri2(i,j)>56
            flag = flag + 1;
        end
    end
    if flag > 2
        delos = [delos, i];
    end
end
tri2(delos,:) = [];

figure;
trisurf(tri1,x1,y1,v11(:,1),'FaceColor','interp')
hold on
trisurf(tri2,x2,y2,v22(:,1),'FaceColor','interp')
xlabel('x')
ylabel('y')
colorbar
title('case4')

rmsv1 = sqrt(sum((v11(:,1) - 10.*v11(:,2)).^2)/size(v11,1))
rmsv2 = sqrt(sum((v22(:,1) - 10.*v22(:,2)).^2)/size(v22,1))