// -*- mode: c++; fill-column: 80; c-basic-offset: 2; indent-tabs-mode: nil -*-
/*
    Copyright (c) 2010 Juan Jose Garcia Ripoll

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


#ifndef TENSOR_JOBS_H
#define TENSOR_JOBS_H

#include <string>
#include <vector>
#include <tensor/sdf.h>

namespace jobs {

  class Job {

  public:
    Job(int argc, const char **argv);

    tensor::index this_job() const;
    tensor::index number_of_jobs() const { return _number_of_jobs; };
    double get_value(const std::string &variable) const;
    void dump_variables(sdf::OutDataFile &file) const;
    void select_job(tensor::index which);

  private:
    class Variable {
    public:
      Variable(const std::string name, double min, double max, tensor::index n = 10) :
        _name(name), _values(tensor::linspace(min, max, n)), _which(0)
      {}
      Variable() :
        _name(""), _values(), _which(0)
      {}

      const std::string &name() const { return _name; }
      const tensor::RTensor &values() const { return _values; }
      const tensor::index size() const { return _values.size(); }
      void select(tensor::index i) { _which = i; }
      double value() const { return _values[_which]; }

    private:
      std::string _name;
      tensor::RTensor _values;
      tensor::index _which;
    };

    static const Variable parse_line(const std::string &s);
    static int parse_file(std::istream &s, std::vector<Variable> &data);
    static const Variable no_variable;

    std::string _filename;
    std::vector<Variable> _variables;
    tensor::index _number_of_jobs;
    tensor::index _this_job;
  };

} // namespace jobs

#endif // ! TENSOR_JOBS_H
