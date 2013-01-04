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

#ifndef TENSOR_SDF_H
#define TENSOR_SDF_H

#include <fstream>
#include <string>
#include <vector>
#include <tensor/tensor.h>

namespace sdf {

  using namespace tensor;

  class DataFile {
  public:
    enum file_tags {
      TAG_RTENSOR = 0,
      TAG_CTENSOR = 1,
      TAG_RTENSOR_VECTOR = 2,
      TAG_CTENSOR_VECTOR = 3
    };

    enum endianness {
      BIG_ENDIAN_FILE = 0,
      LITTLE_ENDIAN_FILE = 1
    };

  protected:
    static const size_t var_name_size = 64;
    std::string _filename;
    std::string _lock_filename;
    int _lock;
    bool _open;
#ifdef TENSOR_BIGENDIAN
    static const enum endianness endian = BIG_ENDIAN_FILE;
#else
    static const enum endianness endian = LITTLE_ENDIAN_FILE;
#endif

    DataFile(const std::string &a_filename, bool lock);
    ~DataFile();
    const char *tag_to_name(size_t tag);
    void close();
    bool is_open() { return _open; }
    bool is_locked() { return _lock; }
  };


  class OutDataFile : public DataFile {

  public:

    OutDataFile(const std::string &a_filename, bool lock = true);
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
    void write_raw(const long *data, size_t n);
    void write_raw(const size_t *data, size_t n);
    void write_raw(const double *data, size_t n);
    void write_raw(const cdouble *data, size_t n);

    template<typename t> void write_raw(t v) {
	write_raw(&v, 1);
    }

    template<class Vector> void dump_vector(const Vector &v);

    void write_header();
    void write_variable_name(const std::string &name);
    void write_tag(const std::string &name, size_t tag);
  };

  class InDataFile : public DataFile {

  public:

    InDataFile(const std::string &a_filename, bool lock = true);

    void load(int &r, const std::string &name = "");
    void load(size_t &r, const std::string &name = "");
    void load(double &r, const std::string &name = "");
    void load(cdouble &r, const std::string &name = "");
    void load(RTensor &t, const std::string &name = "");
    void load(CTensor &t, const std::string &name = "");
    void load(std::vector<RTensor> &m, const std::string &name = "");
    void load(std::vector<CTensor> &m, const std::string &name = "");

    void close();

  private:

    std::ifstream _stream;

    void read_raw(char *data, size_t n);
    void read_raw(int *data, size_t n);
    void read_raw(size_t *data, size_t n);
    void read_raw(long *data, size_t n);
    void read_raw(double *data, size_t n);
    void read_raw(cdouble *data, size_t n);

    template<typename t> void read_raw(t &v) {
	read_raw(&v, 1);
    }

    template<class Vector> const Vector load_vector();

    size_t read_tag_code();
    std::string read_variable_name();

    void read_header();
    void read_tag(const std::string &record_name, size_t tag);
  };

} // namespace sdf

#endif /* !__MPS_IO_H */
