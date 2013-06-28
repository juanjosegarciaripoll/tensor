function record = sdf_sort_1d(record, varargin)
names = fieldnames(record);
data = [];
for i = 1:length(varargin)
    aux = getfield(record, varargin{i});
    data = [data, aux(:)];
end
[~, order] = sortrows(data);
l = length(order);
for i = 1:length(names)
    data = getfield(record, names{i});
    % find the last index in this tensor which is not 1
    dims = size(data);
    k = find(dims > 1);
    k = k(end);
    if dims(k) ~= l
        warn(['The last dimension of field ' names{i} ' does not ' ...
              'match the size of the sort field ' field]);
    else
        data = reshape(data, numel(data)/l, l);
        data = reshape(data(:,order), dims);
        record = setfield(record, names{i}, data);
    end
end

