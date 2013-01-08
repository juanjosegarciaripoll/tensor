function record = sdf_sort_1d(record, field)
names = fieldnames(record);
data = getfield(record, field);
[~, order] = sort(data);
l = length(order);
for i = 1:length(names)
    data = getfield(record, names{i});
    % find the last index in this tensor which is not 1
    dims = size(data);
    k = find(dims > 1);
    k = k(end);
    if dims(k) ~= l
        error(['The last dimension of field ' names{i} ' does not ' ...
               'match the size of the sort field ' field]);
    end
    data = reshape(data, numel(data)/l, l);
    data = reshape(data(:,order), dims);
    record = setfield(record, names{i}, data);
end

