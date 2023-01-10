import sys
import numpy as np
import pathlib

def endian_char(byteorder):
    if byteorder == 'little':
        return '<'
    else:
        return '>'

def load(filename,**kwargs):
    sdf = SDF(filename)
    if 'into' in kwargs:
        return sdf.load_into(kwargs['into'])
    else:
        return sdf.load()

def load_sorted_dataset(path,field):
    set = [load(f) for f in pathlib.Path(path).iterdir()]
    set.sort(key=lambda data: data[field][0])
    return combine_dataset(set)

def load_dataset(path):
    set = [load(f) for f in pathlib.Path(path).iterdir()]
    return combine_dataset(set)

def combine_dataset(set):
    output = {}
    for name in list(set[0].keys()):
        objs = [data[name] for data in set]
        if type(objs[0]) is list:
            output[name] = objs
        else:
            output[name] = np.stack(objs, axis=objs[0].ndim)
    return output

class SDF:
    """Reading binary files from libtensor"""
    # Python array code for int32
    int32 = endian_char(sys.byteorder)+'i4'
    # Python array code for int64
    int64 = endian_char(sys.byteorder)+'i8'
    # Python array code for doubles
    double = endian_char(sys.byteorder)+'d'
    # Python array code for complex doubles
    cdouble = endian_char(sys.byteorder)+'c16'

    def __init__(self, filename, ignored_fields={}):
        self.f = []
        self.filename = pathlib.Path(filename)
        self.long_type = self.int32
        self.endian = sys.byteorder
        self.interpret = False # is the same endian?
        self.ignored_fields = ignored_fields

    def load(self):
        self.f = self.filename.open('rb')
        output = {}
        while self.f.readable():
            obj, name = self.load_record()
            if name is None:
                continue
            if len(name):
                output[name] = obj
            else:
                break
        self.f.close()
        f = []
        return output

    def set_endian(self, newendian):
        self.endian = newendian
        self.interpret = sys.byteorder != newendian

    def load_record(self, skip=False):
        name, code = self.load_tag()
        if name in self.ignored_fields:
            name = None
            skip = True
        obj = []
        if code == -1:
            name = '';
            obj = [];
        elif code == 0:
            obj = self.load_tensor(False, skip)
        elif code == 1:
            obj = self.load_tensor(True, skip)
        elif code == 2:
            obj = self.load_mp(False, skip)
        elif code == 3:
            obj = self.load_mp(True, skip)
        else:
            raise Error('Unknown SDF tag')
        return obj, name

    def load_mp(self, iscomplex, skip):
        L = self.read_longs(1)[0]
        if skip:
            for i in range(L):
                self.load_record(skip)
            return None
        else:
            return [self.load_record()[0] for i in range(L)]

    def load_tensor(self, iscomplex, skip):
        rank = self.read_longs(1)[0]
        dims = self.read_longs(rank)
        L = self.read_longs(1)[0]
        if skip:
            return self.skip_doubles(2 * L if iscomplex else L)
        elif iscomplex:
            data = self.read_complex(L)
        else:
            data = self.read_doubles(L)
        return np.ndarray(shape=dims, buffer=data, dtype=data.dtype, order='F')

    def read_longs(self, n):
        output = np.ndarray(shape=(n,), dtype=self.long_type)
        self.f.readinto(output)
        if self.interpret:
            output.byteswwap()
        return output

    def read_doubles(self, n):
        output = np.ndarray(shape=(n,), dtype=self.double)
        self.f.readinto(output)
        if self.interpret:
            output.byteswwap()
        return output

    def read_complex(self, n):
        output = np.ndarray(shape=(n,), dtype=self.cdouble)
        self.f.readinto(output)
        if self.interpret:
            output.byteswwap()
        return output

    def skip_doubles(self, n):
        self.f.seek(n * 8, 1)
        return None

    def load_tag(self):
        name = self.f.read(64)
        code = []
        if len(name):
            if name[0] == 115 and name[1] == 100 and name[2] == 102:
                if name[4] == ord('4'):
                    self.long_type = self.int32
                elif name[4] == ord('8'):
                    self.long_type = self.int64
                else:
                    raise ValueError('Unsupported tag size in SDF file');
                if name[5] == ord('1'):
                    self.set_endian('little')
                else:
                    self.set_endian('big')
                name = self.f.read(64)
            name, sep, rest = name.partition(b'\x00')
            name = str(name,'utf-8')
            code = self.read_longs(1)[0]
        else:
            code = -1
        return name, code
