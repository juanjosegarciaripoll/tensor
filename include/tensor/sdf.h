#pragma once
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

#ifndef TENSOR_SDF_H
#define TENSOR_SDF_H

#include <fstream>
#include <string>
#include <vector>
#include <tensor/tensor.h>

namespace sdf {

using namespace tensor;

bool file_exists(const std::string &filename);
bool delete_file(const std::string &filename);
bool rename_file(const std::string &orig, const std::string &dest,
                 bool overwrite = true);
bool isdir(const std::string &filename);
bool make_directory(const std::string &dirname, int mode = 0777);

class DataFile {
 public:
  enum file_tags {
    TAG_RTENSOR = 0,
    TAG_CTENSOR = 1,
    TAG_RTENSOR_VECTOR = 2,
    TAG_CTENSOR_VECTOR = 3
  };

  enum endianness { BIG_ENDIAN_FILE = 0, LITTLE_ENDIAN_FILE = 1 };

  enum flags {
    SDF_SHARED = 1,    /* Append with exclusive access */
    SDF_OVERWRITE = 0, /* Overwrite original file */
    SDF_PARANOID = 4   /* Overwrite only when all operations have finished */
  };

 protected:
  const char *_suffix;
  std::string _actual_filename;
  std::string _filename;
  std::string _lock_filename;
  int _flags;
  int _lock;
  bool _open;
  static const enum endianness endian;
  static const unsigned int var_name_size;

  explicit DataFile(const std::string &a_filename, int flags = SDF_SHARED);
  ~DataFile();
  const char *tag_to_name(tensor::index tag);
  void close();
  bool is_open() const { return _open; }
  bool is_locked() const { return _lock; }
  const std::string &actual_filename() const { return _actual_filename; }
};

class OutDataFile : public DataFile {
 public:
  explicit OutDataFile(const std::string &a_filename, int flags = SDF_SHARED);
  ~OutDataFile();

  void dump(const int r, const std::string &name = "");
  void dump(const size_t r, const std::string &name = "");
  void dump(const double r, const std::string &name = "");
  void dump(const cdouble r, const std::string &name = "");
  void dump(const RTensor &t, const std::string &name = "");
  void dump(const CTensor &t, const std::string &name = "");
  void dump(const std::vector<RTensor> &t, const std::string &name = "");
  void dump(const std::vector<CTensor> &t, const std::string &name = "");

  void close();

 private:
  std::ofstream _stream;

  void write_raw(const char *data, size_t n);
  void write_raw(const int *data, size_t n);
  void write_raw(const tensor::index *data, size_t n);
  void write_raw(const size_t *data, size_t n);
  void write_raw(const double *data, size_t n);
  void write_raw(const cdouble *data, size_t n);

  template <typename t>
  void write_raw(t v) {
    write_raw(&v, 1);
  }

  template <class Vector>
  void dump_vector(const Vector &v);

  template <typename iterator>
  void dump_sequence(iterator begin, size_t howmany);

  void write_header();
  void write_variable_name(const std::string &name);
  void write_tag(const std::string &name, tensor::index tag);
};

class InDataFile : public DataFile {
 public:
  explicit InDataFile(const std::string &a_filename, int flags = SDF_SHARED);

  void load(int *r, const std::string &name = "");
  void load(size_t *r, const std::string &name = "");
  void load(double *r, const std::string &name = "");
  void load(cdouble *r, const std::string &name = "");
  void load(RTensor *t, const std::string &name = "");
  void load(CTensor *t, const std::string &name = "");
  void load(std::vector<RTensor> *m, const std::string &name = "");
  void load(std::vector<CTensor> *m, const std::string &name = "");

  void close();

 private:
  std::ifstream _stream;

  void read_raw(char *data, size_t n);
  void read_raw(int *data, size_t n);
  void read_raw(size_t *data, size_t n);
  void read_raw(tensor::index *data, size_t n);
  void read_raw(double *data, size_t n);
  void read_raw(cdouble *data, size_t n);

  template <typename t>
  void read_raw(t &v) {
    read_raw(&v, 1);
  }

  template <class Vector>
  const Vector load_vector();

  tensor::index read_tag_code();
  std::string read_variable_name();

  void read_header();
  void read_tag(const std::string &record_name, tensor::index tag);
};

}  // namespace sdf

#endif /* !__MPS_IO_H */
