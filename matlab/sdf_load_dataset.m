function [varargout] = sdf_load_dataset(directory, varargin)
global SDF_TAG_TYPE SDF_DATA;
if isempty(SDF_TAG_TYPE)
  SDF_TAG_TYPE='int32';
end;
list = dir(directory);
k = find(directory == '/');
if ~isempty(k)
    directory = directory(1:(k(end)-1));
end
SDF_DATA = [];
for i = 1:length(list)
    if ~list(i).isdir
        sdf_load_one([directory '/' list(i).name]);
    end
end
if nargin == 1
  varargout{1} = SDF_DATA;
else
  for i = 1:min(nargout, length(varargin))
    f = varargin{i};
    if ~isfield(SDF_DATA, f)
      error(['Variable ' f ' was not found in file ' filename]);
    end;
    varargout{i} = getfield(SDF_DATA,f);
  end;
end;


function sdf_load_one(filename)
global SDF_DATA;
f = {fopen(filename, 'rb'), filename};
if f{1} >= 0
  %try
    while ~feof(f{1})
      [obj,name,dims] = sdf_load_record(f);
      if isempty(name)
        if isempty(obj)
          break;
        end;
      else
        if isfield(SDF_DATA, name)
          old = getfield(SDF_DATA, name);
          l = length(dims);
          if l > 1 || size(obj,1) > 1
              l = l+1;
          end
          obj = cat(l, old, obj);
        end;
        SDF_DATA = setfield(SDF_DATA, name, obj);
      end;
    end;
    fclose(f{1});
  %catch
  %  fclose(f{1});
  %  rethrow(lasterror);
  %end;
else
    error(['Cannot open file ' filename]);
end;
