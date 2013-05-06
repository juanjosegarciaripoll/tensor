function [obj,name,dims] = sdf_load_record(f)
[name, code] = sdf_load_tag(f);
switch code
 case -1
  name = '';
  obj = [];
  dims = [length(name)];
 case 0
  [obj,dims] = sdf_load_tensor(f, 0);
 case 1
  [obj,dims] = sdf_load_tensor(f, 1);
 case 2
  obj = sdf_load_mp(f, 0);
  dims = [length(obj)];
 case 3
  obj = sdf_load_mp(f, 1);
  dims = [length(obj)];
 otherwise
  error(['Wrong tag, ' num2str(code) ', found while reading ' f{2}]);
end


function v = read_longs(fid, n)
global SDF_TAG_TYPE;
global SDF_ENDIAN;
v = fread(fid, n, SDF_TAG_TYPE,0, SDF_ENDIAN);


function [name,code] = sdf_load_tag(f, expected)
global SDF_TAG_TYPE;
global SDF_ENDIAN;
name = fread(f{1}, 64, 'char');
if ~isempty(name) && all(name(1:3) == [115; 100; 102]) % sdf
  if name(5) == '4'
    SDF_TAG_TYPE='int32';
  elseif name(5) == '8'
    SDF_TAG_TYPE='int64';
  else
    error(['Unsupported tag size ', name(5)]);
  end;
  if name(6) == '1'
    SDF_ENDIAN = 'b';
  else
    SDF_ENDIAN = 'l';
  end
  name = fread(f{1}, 64, 'char');
end;
if feof(f{1})
  name = '';
  code = -1;
else
  code = read_longs(f{1}, 1);
  name = char(name(find(name ~= 0)));
end;
if nargin == 2 && (code ~= expected)
  error(['Found object of type ' code ' instead of ' expected
         ' while reading ' f{2}]);
end;
name = name(:)';


function [Pk,dims] = sdf_load_tensor(f, cplx)
global SDF_ENDIAN;
rank = read_longs(f{1}, 1);
dims = read_longs(f{1}, rank)';
L    = read_longs(f{1}, 1);
if cplx
  [Pk,size] = fread(f{1}, L*2, 'double', 0, SDF_ENDIAN);
  Pk = complex(Pk(1:2:end),Pk(2:2:end));
else
  Pk = fread(f{1}, L, 'double', 0, SDF_ENDIAN);
end;
if rank > 1
  Pk = reshape(Pk, dims);
end;


function P = sdf_load_mp(f, cplx)
global SDF_LOAD_MPS;
L = read_longs(f{1}, 1);
if isempty(SDF_LOAD_MPS)
  for k = 1:L
    sdf_load_record(f);
  end;
  P = [];
else
  P = create_mp(L);
  for k = 1:L
    set_matrix(sdf_load_record(f), P, k);
  end;
end;
