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

#include <array>
#include <limits>
#include <cstring>
#include <limits>
#include <tensor/sdf.h>

using namespace sdf;

static std::streamsize safe_streamsize(size_t size) {
  tensor_assert2(
      size <= static_cast<size_t>(std::numeric_limits<std::streamsize>::max()),
      std::overflow_error("SDF record exceeds std::streamsize"));
  return static_cast<std::streamsize>(size);
}

#if !defined(aix)

template <class number>
void write_raw_with_endian(std::ofstream &s, const number *data, size_t n) {
  s.write(reinterpret_cast<const char *>(data), // NOLINT
          safe_streamsize(n * sizeof(number)));
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
    s.write(reinterpret_cast<const char *>(data), // NOLINT
            safe_streamsize(n * sizeof(number)));
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
    s.write(buffer, safe_streamsize(now));
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

OutDataFile::OutDataFile(std::string a_filename, int a_flags)
  : DataFile(std::move(a_filename), a_flags) {
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
  tensor_assert(is_open());
  write_raw_with_endian(_stream, data, n);
}

void OutDataFile::write_raw(const int *data, size_t n) {
  tensor_assert(is_open());
  write_raw_with_endian(_stream, data, n);
}

void OutDataFile::write_raw(const index_t *data, size_t n) {
  tensor_assert(is_open());
  write_raw_with_endian(_stream, data, n);
}

void OutDataFile::write_raw(const size_t *data, size_t n) {
  tensor_assert(is_open());
  write_raw_with_endian(_stream, data, n);
}

void OutDataFile::write_raw(const double *data, size_t n) {
  tensor_assert(is_open());
  write_raw_with_endian(_stream, data, n);
}

void OutDataFile::write_raw(const cdouble *data, size_t n) {
  tensor_assert(is_open());
  write_raw_with_endian(_stream, reinterpret_cast<const double *>(data), 2 * n); // NOLINT
}

void OutDataFile::write_variable_name(const std::string &name) {
  std::string buffer;
  buffer.assign(var_name_size, static_cast<char>(0));
  buffer.replace(0, std::min<size_t>(var_name_size - 1, name.size()), name);
  write_raw(buffer.c_str(), var_name_size);
}

void OutDataFile::write_tag(const std::string &name, index_t tag) {
  write_variable_name(name);
  write_raw(tag);
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
  dump_sequence(t.dimensions().begin(), static_cast<size_t>(t.rank()));
  dump_vector(t);
}

void OutDataFile::dump(const CTensor &t, const std::string &name) {
  write_tag(name, TAG_CTENSOR);
  dump_sequence(t.dimensions().begin(), static_cast<size_t>(t.rank()));
  dump_vector(t);
}

void OutDataFile::dump(const std::vector<RTensor> &t, const std::string &name) {
  write_tag(name, TAG_RTENSOR_VECTOR);
  write_raw(t.size());
  for (const auto &value : t) {
    dump(value);
  }
}

void OutDataFile::dump(const std::vector<CTensor> &t, const std::string &name) {
  write_tag(name, TAG_CTENSOR_VECTOR);
  write_raw(t.size());
  for (const auto &value : t) {
    dump(value);
  }
}

void OutDataFile::dump(const double value, const std::string &name) {
  RTensor t({value});
  dump(t, name);
}

void OutDataFile::dump(const cdouble value, const std::string &name) {
  CTensor t({value});
  dump(t, name);
}

void OutDataFile::dump(size_t value, const std::string &name) {
  dump(static_cast<double>(value), name);
}

void OutDataFile::dump(int value, const std::string &name) {
  dump(static_cast<double>(value), name);
}

void OutDataFile::write_header() {
  std::array<char, 7> tag = {'s',
                             'd',
                             'f',
                             sizeof(int) + '0',
                             sizeof(index_t) + '0',
                             (endian == BIG_ENDIAN_FILE) ? '0' : '1',
                             0};

  write_variable_name(&tag[0]);
}
