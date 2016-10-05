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

    def __init__(self, filename):
        self.f = []
        self.filename = pathlib.Path(filename)
        self.long_type = self.int32
        self.endian = sys.byteorder
        self.interpret = False # is the same endian?

    def load(self):
        self.f = self.filename.open("rb")
        output = {}
        while self.f.readable():
            obj, name = self.load_record()
            if not name:
                break
            output[name] = obj
        self.f.close()
        f = []
        return output

    def set_endian(self, newendian):
        self.endian = newendian
        self.interpret = sys.byteorder != newendian

    def load_record(self):
        name, code = self.load_tag()
        obj = []
        if name:
            code = code[0]
            if code == -1:
                name = '';
                obj = [];
            elif code == 0:
                obj = self.load_tensor(False)
            elif code == 1:
                obj = self.load_tensor(True)
            else:
                raise Error('Unknown SDF tag')
        return obj, name

    def load_mp(self, iscomplex):
        [ self.load_tensor(iscomplex) for i in range(sefl.read_longs(1))]

    def load_tensor(self, iscomplex):
        rank = self.read_longs(1)
        dims = self.read_longs(rank)
        L = self.read_longs(1)
        if iscomplex:
            data = self.read_complex(L)
        else:
            data = self.read_doubles(L)
        return np.ndarray(shape=dims,buffer=data,dtype=data.dtype,order='F')

    def read_longs(self, n):
        output = np.ndarray(shape=(n,),dtype=self.long_type)
        self.f.readinto(output)
        if self.interpret:
            output.byteswwap()
        return output

    def read_doubles(self, n):
        output = np.ndarray(shape=(n,),dtype=self.double)
        self.f.readinto(output)
        if self.interpret:
            output.byteswwap()
        return output
    
    def read_complex(self, n):
        output = np.ndarray(shape=(n,),dtype=self.cdouble)
        self.f.readinto(output)
        if self.interpret:
            output.byteswwap()
        return output

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
            code = self.read_longs(1)
        return name, code
