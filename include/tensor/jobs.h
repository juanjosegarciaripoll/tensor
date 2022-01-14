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

#pragma once
#ifndef TENSOR_JOBS_H
#define TENSOR_JOBS_H

#include <string>
#include <vector>
#include <tensor/refcount.h>
#include <tensor/sdf.h>

namespace jobs {

class Job {
 public:
  Job(int argc, const char **argv);

  tensor::index number_of_jobs() const { return number_of_jobs_; };
  tensor::index current_job() const { return this_job_; };
  void operator++();
  bool to_do();

  double get_value_with_default(const std::string &variable, double def) const;
  double get_value(const std::string &variable) const;
  void dump_variables(sdf::OutDataFile &file) const;

  typedef tensor::shared_ptr<sdf::OutDataFile> dataset;
  dataset open_dataset(const std::string &filename) const;
  bool dataset_record_exists(const std::string &filename) const;

 private:
  class Variable {
   public:
    Variable(const std::string name, double min, double max,
             tensor::index n = 10)
        : name_(name), values_(tensor::linspace(min, max, n)), which_(0) {}
    Variable() : name_(""), values_(), which_(0) {}

    const std::string &name() const { return name_; }
    const tensor::RTensor &values() const { return values_; }
    tensor::index size() const { return values_.size(); }
    void select(tensor::index i) { which_ = i; }
    double value() const { return values_[which_]; }

   private:
    std::string name_;
    tensor::RTensor values_;
    tensor::index which_;
  };

  typedef std::vector<Variable> var_list;

  static const Variable no_variable;
  std::string filename_;
  var_list variables_;
  tensor::index number_of_jobs_;
  tensor::index this_job_, first_job_, last_job_;

  const Variable *find_variable(const std::string &name) const;
  static const Variable parse_line(const std::string &s);
  static int parse_file(std::istream &s, std::vector<Variable> &data);
  tensor::index compute_number_of_jobs() const;
  void select_job(tensor::index which);
};

}  // namespace jobs

#endif  // ! TENSOR_JOBS_H
