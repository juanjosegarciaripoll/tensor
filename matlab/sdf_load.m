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
          obj = cat(length(dims)+1, old, obj);
        end;
        data = setfield(data, name, obj);
      end;
    end;
    fclose(f{1});
  %catch
  %  fclose(f{1});
  %  rethrow(lasterror);
  %end;
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
