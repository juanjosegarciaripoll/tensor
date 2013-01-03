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

#include <tensor/sdf.h>

using namespace sdf;

#if !defined(aix)
template <class number>
void read_no_error_with_endian(std::ifstream &s, number *data, size_t n)
{
  s.read((char *)data, n * sizeof(number));
  if (s.bad()) {
    std::cerr << "I/O error when reading from stream " << s;
    abort();
  }
}
#else
template <class number>
void read_no_error_with_endian(std::ifstream &s, number *data, size_t n)
{
  const int size = sizeof(number);
  if (size == 1) {
    s.read((char *)data, n * sizeof(number));
    if (s.bad()) {
      std::cerr << "I/O error when reading from stream " << s;
      abort();
    }
    return;
  }

  const int buffer_size = 1024;
  char *alias = (char *)data;
  char buffer[buffer_size];

  do {
    size_t now = min<size_t>(n*size, buffer_size);
    s.read(buffer, now);
    if (s.bad()) {
      std::cerr << "I/O error when reading from stream " << s;
      abort();
    }
    for (size_t i = 0; i < now; i+=size) {
      if (size == 4) {
	*(alias++) = buffer[i+3];
	*(alias++) = buffer[i+2];
	*(alias++) = buffer[i+1];
	*(alias++) = buffer[i];
      } else if (size == 8) {
	*(alias++) = buffer[i+7];
	*(alias++) = buffer[i+6];
	*(alias++) = buffer[i+5];
	*(alias++) = buffer[i+4];
	*(alias++) = buffer[i+3];
	*(alias++) = buffer[i+2];
	*(alias++) = buffer[i+1];
	*(alias++) = buffer[i];
      } else {
	for (size_t j = size; j--; ) {
	  *(alias++) = buffer[i+j];
	}
      }
      n--;
    }
  } while (n);
}
#endif

InDataFile::InDataFile(const std::string &a_filename) :
  DataFile(a_filename), _stream(a_filename.c_str())
{
  read_header();
}

void
InDataFile::read_no_error(char *data, size_t n)
{
  read_no_error_with_endian(_stream, data, n);
}

void
InDataFile::read_no_error(int *data, size_t n)
{
  read_no_error_with_endian(_stream, data, n);
}

void
InDataFile::read_no_error(long *data, size_t n)
{
  read_no_error_with_endian(_stream, data, n);
}

void
InDataFile::read_no_error(size_t *data, size_t n)
{
  read_no_error_with_endian(_stream, data, n);
}

void
InDataFile::read_no_error(double *data, size_t n)
{
  read_no_error_with_endian(_stream, data, n);
}

void
InDataFile::read_no_error(cdouble *data, size_t n)
{
  read_no_error_with_endian(_stream, (double*)data, 2*n);
}

size_t
InDataFile::read_tag_code()
{
  size_t output;
  read_no_error(output);
  return output;
}

std::string
InDataFile::read_variable_name()
{
  char buffer[var_name_size];
  read_no_error(buffer, var_name_size);
  return std::string(buffer);
}

void
InDataFile::read_tag(const std::string &name, size_t type)
{
  std::string other_name = read_variable_name();
  if (name.size() && (name != other_name)) {
    std::cerr << "While reading file " << _filename << ", variable "
	      << name << " was expected but found "
	      << other_name << '\n';
    abort();
  }
  size_t other_type = read_tag_code();
  if (type != other_type) {
    std::cerr << "While reading file " << _filename << ", an object of type "
	      << tag_to_name(type) << " was expected but found a "
	      << tag_to_name(other_type) << '\n';
    abort();
  }
}

template<class Vector>
const Vector InDataFile::load_vector()
{
  size_t length;
  read_no_error(length);
  Vector v(length);
  read_no_error(v.begin(), length);
  return v;
}

void
InDataFile::load(RTensor &t, const std::string &name) {
  read_tag(name, TAG_RTENSOR);
  Indices dims = load_vector<Indices>();
  t = RTensor(dims, load_vector<RTensor>());
}

void
InDataFile::load(CTensor &t, const std::string &name) {
  read_tag(name, TAG_CTENSOR);
  Indices dims = load_vector<Indices>();
  t = CTensor(dims, load_vector<RTensor>());
}

void
InDataFile::load(std::vector<RTensor> &m, const std::string &name)
{
  read_tag(name, TAG_RTENSOR_VECTOR);
  size_t l;
  read_no_error(l);
  m.resize(l);
  for (size_t k = 0; k < l; k++) {
    load(m.at(k));
  }
}

void
InDataFile::load(std::vector<CTensor> &m, const std::string &name)
{
  read_tag(name, TAG_CTENSOR_VECTOR);
  size_t l;
  read_no_error(l);
  m.resize(l);
  for (size_t k = 0; k < l; k++) {
    load(m.at(k));
  }
}

void
InDataFile::load(double &value, const std::string &name)
{
  RTensor t;
  load(t, name);
  if (t.size() > 1) {
    std::cerr << "While reading file " << _filename << " found a tensor of size "
	      << t.size() << " while a single value was expected.";
    abort();
  }
  value = t[0];
}

void
InDataFile::load(cdouble &value, const std::string &name)
{
  CTensor t;
  load(t, name);
  if (t.size() > 1) {
    std::cerr << "While reading file " << _filename << " found a tensor of size "
	      << t.size() << " while a single value was expected.";
    abort();
  }
  value = t[0];
}

void
InDataFile::load(size_t &v, const std::string &name)
{
  double aux = v;
  load(aux, name);
  v = (size_t)aux;
}

void
InDataFile::load(int &v, const std::string &name)
{
  double aux = v;
  load(aux, name);
  v = (int)aux;
}

void
InDataFile::read_header()
{
  std::string var_name = read_variable_name();

  if (var_name[0] != 's' || var_name[1] != 'd' || var_name[2] != 'f') {
    std::cerr << "Bogus SDF file";
    abort();
  }
  int file_int_size = var_name[3] - '0';
  int file_long_size = var_name[4] - '0';
  int file_endianness = var_name[5] - '0';
  if (file_int_size != sizeof(int) ||
      file_long_size != sizeof(long) ||
      file_endianness != endian)
    {
      std::cerr << "File " << _filename << " has word sizes (" << file_int_size
		<< ',' << file_long_size << ") and cannot be read by this computer";
      abort();
    }
}
