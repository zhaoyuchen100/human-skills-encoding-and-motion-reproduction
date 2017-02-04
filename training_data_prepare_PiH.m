%% training data prepare
function [GMM]=training_data_prepare_PiH(data,curser_indx,normalization_flag,varargin)
if nargin >3
mu = varargin(1);
dev = varargin(2);
apply_norm = 1;
else 
    apply_norm = 0;
end
if size(data,1)>size(data,2)
    data = data';
end
for i = 1:length(curser_indx)
indx(i) = curser_indx(i).Position(1);
end
indx = sort(indx);
ind.group1_tmp = [];
timeline = [];
timeline_old = [];
tmp = [];
cat_data_indx = [];
for i = 1:length(curser_indx)/2
    cat_data_indx = [cat_data_indx,indx(2*(i-1)+1):indx(2*(i-1)+2)];
end
if normalization_flag
     [GMM.orig_data(2:size(data,1)+1,:),GMM.mean,GMM.dev] = zscore([data(:,cat_data_indx)],1,2);
    else
     GMM.orig_data(2:size(data,1)+1,:) = [data(:,cat_data_indx)];
end

for i = 1:length(indx)
if 2*i>length(indx)
    break;
end
    interval1 = indx(2*(i-1)+1):indx(2*i);
if normalization_flag
    data_group1{i} = [(data(:,interval1)-repmat(GMM.mean(:),[1,length(interval1)]))./repmat(GMM.dev(:),[1,length(interval1)])];
    ind.group1 = [ind.group1_tmp,interval1];
elseif apply_norm
    data_group1{i} = [(data(:,interval1)-repmat(mu{1}(:),[1,length(interval1)]))./repmat(dev{1}(:),[1,length(interval1)])];
    ind.group1 = [ind.group1_tmp,interval1];
else
    data_group1{i} = [data(:,interval1)];
    ind.group1 = [ind.group1_tmp,interval1];
end
    ind.group1_tmp = ind.group1;
    timeline = horzcat(timeline_old,1:length(data_group1{i}));
    timeline_old = timeline;
end
GMM.orig_data(1,:) = timeline;
GMM.data_cell = data_group1;
end