function [varargout] = sdf_load(filename, varargin)
global SDF_TAG_TYPE;
if isempty(SDF_TAG_TYPE)
  SDF_TAG_TYPE='int32';
end;
f = {fopen(filename, 'rb'), filename};
data = [];
if f{1} >= 0
  %try
    while ~feof(f{1})
      [obj,name,dims] = sdf_load_record(f);
      if isempty(name)
        if isempty(obj)
          break;
        end;
      else
        if isfield(data, name)
          old = getfield(data, name);
          l = length(dims);
          if l > 1 || size(obj,1) > 1
              l = l+1;
          end
          obj = cat(l, old, obj);
        end;
        data = setfield(data, name, obj);
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
if nargin == 1
  varargout{1} = data;
else
  for i = 1:min(nargout, length(varargin))
    f = varargin{i};
    if ~isfield(data, f)
      error(['Variable ' f ' was not found in file ' filename]);
    end;
    varargout{i} = getfield(data,f);
  end;
end;
