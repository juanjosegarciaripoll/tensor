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

#include <cstring>
#include <tensor/sdf.h>

using namespace sdf;

#if !defined(aix)

template <class number>
void write_raw_with_endian(std::ofstream &s, const number *data, size_t n) {
  s.write((char *)data, n * sizeof(number));
  if (s.bad()) {
    std::cerr << "I/O error when writing to SDF stream";
    abort();
  }
  // Horrible hack to ensure icc flushes our buffers and does not
  // cause buffer overflow.
  s.flush();
}

#else

template <class number>
void write_raw_with_endian(std::ofstream &s, const number *data, size_t n) {
  const int size = sizeof(number);
  if (size == 1) {
    s.write((char *)data, n * sizeof(number));
    if (s.bad()) {
      std::cerr << "I/O error when writing to SDF stream";
      abort();
    }
    s.flush();
    return;
  }

  const int buffer_size = 1024;
  const char *alias = (char *)data;
  char buffer[buffer_size];

  do {
    size_t now = std::min<size_t>(n, buffer_size / size) * size;
    for (size_t i = 0; i < now; i += size) {
      if (size == 4) {
        buffer[i + 3] = *(alias++);
        buffer[i + 2] = *(alias++);
        buffer[i + 1] = *(alias++);
        buffer[i] = *(alias++);
      } else if (size == 8) {
        buffer[i + 7] = *(alias++);
        buffer[i + 6] = *(alias++);
        buffer[i + 5] = *(alias++);
        buffer[i + 4] = *(alias++);
        buffer[i + 3] = *(alias++);
        buffer[i + 2] = *(alias++);
        buffer[i + 1] = *(alias++);
        buffer[i] = *(alias++);
      } else {
        for (size_t j = size; j--;) {
          buffer[i + j] = *(alias++);
        }
      }
      n--;
    }
    s.write(buffer, now);
    if (s.bad()) {
      std::cerr << "I/O error when reading from stream " << s;
      abort();
    }
    s.flush();
  } while (n);
}

#endif

//----------------------------------------------------------------------
// COMMON I/O ROUTINES
//

OutDataFile::OutDataFile(const std::string &a_filename, int flags)
    : DataFile(a_filename, flags) {
  bool existed = file_exists(actual_filename());
  _stream.open(actual_filename().c_str(),
               std::ofstream::app | std::ofstream::binary);
  if (!existed) {
    write_header();
  }
}

OutDataFile::~OutDataFile() { close(); }

void OutDataFile::close() {
  if (is_open()) {
    _stream.flush();
    _stream.close();
    DataFile::close();
  }
}

void OutDataFile::write_raw(const char *data, size_t n) {
  assert(is_open());
  write_raw_with_endian(_stream, data, n);
}

void OutDataFile::write_raw(const int *data, size_t n) {
  assert(is_open());
  write_raw_with_endian(_stream, data, n);
}

void OutDataFile::write_raw(const tensor::index *data, size_t n) {
  assert(is_open());
  write_raw_with_endian(_stream, data, n);
}

void OutDataFile::write_raw(const size_t *data, size_t n) {
  assert(is_open());
  write_raw_with_endian(_stream, data, n);
}

void OutDataFile::write_raw(const double *data, size_t n) {
  assert(is_open());
  write_raw_with_endian(_stream, data, n);
}

void OutDataFile::write_raw(const cdouble *data, size_t n) {
  assert(is_open());
  write_raw_with_endian(_stream, (double *)data, 2 * n);
}

void OutDataFile::write_variable_name(const std::string &name) {
  std::string buffer;
  buffer.assign(var_name_size, static_cast<char>(0));
  buffer.replace(0, std::min<size_t>(var_name_size - 1, name.size()), name);
  write_raw(buffer.c_str(), var_name_size);
}

void OutDataFile::write_tag(const std::string &name, tensor::index type) {
  write_variable_name(name);
  write_raw(type);
}

template <class Vector>
void OutDataFile::dump_vector(const Vector &v) {
  OutDataFile::dump_sequence(v.begin(), v.size());
}

template <typename iterator>
void OutDataFile::dump_sequence(iterator begin, size_t howmany) {
  write_raw(howmany);
  write_raw(begin, howmany);
}

void OutDataFile::dump(const RTensor &t, const std::string &name) {
  write_tag(name, TAG_RTENSOR);
  dump_sequence(t.dimensions().begin(), t.rank());
  dump_vector(t);
}

void OutDataFile::dump(const CTensor &t, const std::string &name) {
  write_tag(name, TAG_CTENSOR);
  dump_sequence(t.dimensions().begin(), t.rank());
  dump_vector(t);
}

void OutDataFile::dump(const std::vector<RTensor> &m, const std::string &name) {
  size_t l = m.size();
  write_tag(name, TAG_RTENSOR_VECTOR);
  write_raw(l);
  for (size_t k = 0; k < l; k++) {
    dump(m[k]);
  }
}

void OutDataFile::dump(const std::vector<CTensor> &m, const std::string &name) {
  size_t l = m.size();
  write_tag(name, TAG_CTENSOR_VECTOR);
  write_raw(l);
  for (size_t k = 0; k < l; k++) {
    dump(m[k]);
  }
}

void OutDataFile::dump(const double v, const std::string &name) {
  RTensor t(Dimensions{1}, Vector<double>({v}));
  dump(t, name);
}

void OutDataFile::dump(const cdouble v, const std::string &name) {
  CTensor t(Dimensions{1}, Vector<cdouble>({v}));
  dump(t, name);
}

void OutDataFile::dump(size_t v, const std::string &name) {
  dump((double)v, name);
}

void OutDataFile::dump(int v, const std::string &name) {
  dump((double)v, name);
}

void OutDataFile::write_header() {
  char tag[7] = "sdf  ";
  tag[3] = sizeof(int) + '0';
  tag[4] = sizeof(tensor::index) + '0';
  tag[5] = (endian == BIG_ENDIAN_FILE) ? '0' : '1';

  write_variable_name(tag);
}
