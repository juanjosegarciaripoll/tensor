function record = sdf_sort_2d(record, fieldx, fieldy, rows, cols)
if ~isfield(record, fieldx)
    error([fieldx ' is not a valid field name into the SDF ' ...
           'structure']);
end
if ~isfield(record, fieldy)
    error([fieldy ' is not a valid field name into the SDF ' ...
           'structure']);
end
datax = getfield(record, fieldx);
datay = getfield(record, fieldy);
datax = datax - min(datax);
datay = datay - min(datay);
[~, order] = sort(datax/max(datax) + 2*datay/min(datay(find(datay))));
names = fieldnames(record);
l = numel(datax);
if l ~= numel(datay)
    error(['The fields ' fieldx ' and ' fieldy ' have different ' ...
           'size']);
end
if nargin < 3
    x = datax(1);
    rows = length(find(datax == x));
end
if nargin < 4
    cols = l / rows
    if fix(cols) ~= cols
        error([field ' cannot be reshaped to have ' num2str(rows) ...
               ' rows']);
    end
else
    cols = varargin{1};
    if cols*rows ~= l
        error([field ' does not have the requested shape [' num2str(rows) ...
               ', ' num2str(cols) ']']);
    end
end
new_dims = [rows, cols];
for i = 1:length(names)
    data = getfield(record, names{i});
    dims = size(data);
    k = find(dims > 1);
    k = k(end);
    if dims(k) ~= l
        error(['The tensor ' names{i} ' does not have an index of ' ...
               'size ' num2str(l)]);
    end
    data = reshape(data, [numel(data)/l, l]);
    data = data(:, order);
    data = reshape(data, [dims(1:(k-1)), new_dims]);
    record = setfield(record, names{i}, data);
end
