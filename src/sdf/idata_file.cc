// -*- mode: c++; fill-column: 80; c-basic-offset: 2; indent-tabs-mode: nil -*-
/*
    Copyright (c) 2013 Juan Jose Garcia Ripoll

    Tensor is free software; you can redistribute it and/or modify it
    under the terms of the GNU Library General Public License as published
    by the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Library General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
*/

#include <limits>
#include <tensor/sdf.h>

using namespace sdf;

static std::streamsize safe_streamsize(size_t size) {
  if (size > static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) {
    throw std::overflow_error("Data record too large for std::istream");
  }
  return static_cast<std::streamsize>(size);
}

#if !defined(aix)
template <class number>
void read_raw_with_endian(std::ifstream &s, number *data, size_t n) {
  size_t size = n * sizeof(number);
  s.read(reinterpret_cast<char *>(data), safe_streamsize(size));
  if (s.bad()) {
    std::cerr << "I/O error when reading from SDF stream";
    abort();
  }
}
#else
template <class number>
void read_raw_with_endian(std::ifstream &s, number *data, size_t n) {
  const int size = sizeof(number);
  if (size == 1) {
    s.read((char *)data, safe_streamsize(n * sizeof(number)));
    if (s.bad()) {
      std::cerr << "I/O error when reading from SDF stream";
      abort();
    }
    return;
  }

  const int buffer_size = 1024;
  char *alias = (char *)data;
  char buffer[buffer_size];

  do {
    size_t now = min<size_t>(n * size, buffer_size);
    s.read(buffer, safe_streamsize(now));
    if (s.bad()) {
      std::cerr << "I/O error when reading from stream " << s;
      abort();
    }
    for (size_t i = 0; i < now; i += size) {
      if (size == 4) {
        *(alias++) = buffer[i + 3];
        *(alias++) = buffer[i + 2];
        *(alias++) = buffer[i + 1];
        *(alias++) = buffer[i];
      } else if (size == 8) {
        *(alias++) = buffer[i + 7];
        *(alias++) = buffer[i + 6];
        *(alias++) = buffer[i + 5];
        *(alias++) = buffer[i + 4];
        *(alias++) = buffer[i + 3];
        *(alias++) = buffer[i + 2];
        *(alias++) = buffer[i + 1];
        *(alias++) = buffer[i];
      } else {
        for (size_t j = size; j--;) {
          *(alias++) = buffer[i + j];
        }
      }
      n--;
    }
  } while (n);
}
#endif

InDataFile::InDataFile(const std::string &a_filename, int a_flags)
    : DataFile(a_filename, a_flags),
      _stream(actual_filename().c_str(),
              std::ios_base::in | std::ios_base::binary) {
  _stream.seekg(std::ios_base::beg);
  read_header();
}

void InDataFile::read_raw(char *data, size_t n) {
  assert(is_open());
  read_raw_with_endian(_stream, data, n);
}

void InDataFile::read_raw(int *data, size_t n) {
  assert(is_open());
  read_raw_with_endian(_stream, data, n);
}

void InDataFile::read_raw(tensor::index *data, size_t n) {
  assert(is_open());
  read_raw_with_endian(_stream, data, n);
}

void InDataFile::read_raw(size_t *data, size_t n) {
  assert(is_open());
  read_raw_with_endian(_stream, data, n);
}

void InDataFile::read_raw(double *data, size_t n) {
  assert(is_open());
  read_raw_with_endian(_stream, data, n);
}

void InDataFile::read_raw(cdouble *data, size_t n) {
  assert(is_open());
  read_raw_with_endian(_stream, reinterpret_cast<double *>(data), 2 * n);
}

tensor::index InDataFile::read_tag_code() {
  tensor::index output;
  read_raw(output);
  return output;
}

std::string InDataFile::read_variable_name() {
  char *buffer = new char[var_name_size];
  read_raw(buffer, var_name_size);
  std::string output(buffer);
  delete[] buffer;
  return output;
}

void InDataFile::read_tag(const std::string &name, tensor::index type) {
  std::string other_name = read_variable_name();
  if (name.size() && (name != other_name)) {
    std::cerr << "While reading file " << _filename << ", variable " << name
              << " was expected but found " << other_name << '\n';
    abort();
  }
  tensor::index other_type = read_tag_code();
  if (type != other_type) {
    std::cerr << "While reading file " << _filename << ", an object of type "
              << tag_to_name(type) << " was expected but found a "
              << tag_to_name(other_type) << '\n';
    abort();
  }
}

template <class Vector>
const Vector InDataFile::load_vector() {
  size_t length;
  read_raw(length);
  Vector v(length);
  read_raw(v.begin(), length);
  return v;
}

void InDataFile::load(RTensor *t, const std::string &name) {
  read_tag(name, TAG_RTENSOR);
  Indices dims = load_vector<Indices>();
  *t = RTensor(dims, load_vector<Vector<double>>());
}

void InDataFile::load(CTensor *t, const std::string &name) {
  read_tag(name, TAG_CTENSOR);
  Indices dims = load_vector<Indices>();
  *t = CTensor(dims, load_vector<Vector<cdouble>>());
}

void InDataFile::load(std::vector<RTensor> *m, const std::string &name) {
  read_tag(name, TAG_RTENSOR_VECTOR);
  size_t l;
  read_raw(l);
  m->resize(l);
  for (size_t k = 0; k < l; k++) {
    load(&m->at(k));
  }
}

void InDataFile::load(std::vector<CTensor> *m, const std::string &name) {
  read_tag(name, TAG_CTENSOR_VECTOR);
  size_t l;
  read_raw(l);
  m->resize(l);
  for (size_t k = 0; k < l; k++) {
    load(&m->at(k));
  }
}

void InDataFile::load(double *value, const std::string &name) {
  RTensor t;
  load(&t, name);
  if (t.size() > 1) {
    std::cerr << "While reading file " << _filename
              << " found a tensor of size " << t.size()
              << " while a single value was expected.";
    abort();
  }
  *value = t[0];
}

void InDataFile::load(cdouble *value, const std::string &name) {
  CTensor t;
  load(&t, name);
  if (t.size() > 1) {
    std::cerr << "While reading file " << _filename
              << " found a tensor of size " << t.size()
              << " while a single value was expected.";
    abort();
  }
  *value = t[0];
}

void InDataFile::load(size_t *v, const std::string &name) {
  double aux;
  load(&aux, name);
  *v = static_cast<size_t>(aux);
}

void InDataFile::load(int *v, const std::string &name) {
  double aux;
  load(&aux, name);
  *v = static_cast<int>(aux);
}

void InDataFile::read_header() {
  std::string var_name = read_variable_name();

  if (var_name[0] != 's' || var_name[1] != 'd' || var_name[2] != 'f') {
    std::cerr << "Bogus SDF file" << std::endl
              << "Wrong header: '" << var_name << "'" << std::endl;
    abort();
  }
  int file_int_size = var_name[3] - '0';
  int file_index_size = var_name[4] - '0';
  int file_endianness = var_name[5] - '0';
  if (file_int_size != sizeof(int) ||
      file_index_size != sizeof(tensor::index) || file_endianness != endian) {
    std::cerr << "File " << _filename << " has word sizes (" << file_int_size
              << ',' << file_index_size
              << ") and cannot be read by this computer";
    abort();
  }
}

void InDataFile::close() {
  if (is_open()) {
    DataFile::close();
    _open = false;
    _stream.close();
  }
}
