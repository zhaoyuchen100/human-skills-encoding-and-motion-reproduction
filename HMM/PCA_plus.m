function [X,Coeff_red,explain] = PCA_plus(Y,th,level)
% Y is N by D. D is dimentsion of data!
% each column of ceoeff contains coefficients for one PC
Y = Y' - repmat(mean(Y'),length(Y),1);
if size(Y,1) < size(Y,2)
    Y = Y';
end
[Coeff,score,~,~,explain] = pca(Y);
info = 0;% initialize amount of infomation we want to contain
if th~=0
for i = 1:length(Coeff)
    info = info + explain(i);
    if info >= th
        X = Coeff(:,1:i)'*Y';
        Coeff_red = Coeff(:,1:i);
        break;
    end
end
end
if level~= 0
    X = Coeff(:,1:level)'*Y';
    Coeff_red = Coeff(:,1:level);
end

