#pragma once
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
#include <memory>
#include <tensor/sdf.h>

namespace jobs {

class Job {
 public:
  using dataset = std::shared_ptr<sdf::OutDataFile>;
  using index = tensor::index_t;

  Job() = delete;
  Job(const Job &) = delete;
  Job(Job &&) = delete;
  Job(int argc, const char **argv);

  index number_of_jobs() const { return number_of_jobs_; };
  index current_job() const { return this_job_; };
  void operator++();
  bool to_do() const;

  double get_value_with_default(const std::string &name, double def) const;
  double get_value(const std::string &name) const;
  void dump_variables(sdf::OutDataFile &file) const;

  dataset open_dataset(const std::string &filename) const;
  bool dataset_record_exists(const std::string &filename) const;

 private:
  static constexpr index default_steps = 10;

  class Variable {
   public:
    Variable() = default;
    Variable(std::string name, double min, double max, index n = default_steps)
        : name_(std::move(name)), values_(tensor::linspace(min, max, n)) {}

    const std::string &name() const { return name_; }
    const tensor::RTensor &values() const { return values_; }
    index size() const { return values_.ssize(); }
    void select(index i) { which_ = i; }
    double value() const { return values_[which_]; }

   private:
    std::string name_{};
    tensor::RTensor values_{};
    index which_{};
  };

  using var_list = std::vector<Variable>;

  static const Variable no_variable;
  std::string filename_{};
  var_list variables_{};
  index number_of_jobs_{0};
  index this_job_{0}, first_job_{0}, last_job_{0};

  const Variable *find_variable(const std::string &name) const;
  static const Variable parse_line(const std::string &s);
  static int parse_file(std::istream &s, std::vector<Variable> &data);
  index compute_number_of_jobs() const;
  void select_job(index which);
};

}  // namespace jobs

#endif  // ! TENSOR_JOBS_H
