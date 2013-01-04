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

#include <tensor/jobs.h>
#include <fstream>

using namespace jobs;

static inline
bool is_space(char c)
{
  return (c == ' ') || (c == '\t') || (c == '\n');
}

std::vector<std::string>
split_string(const std::string &s)
{
  std::vector<std::string> output;
  size_t i = 0, l = s.size();
  while (i != l) {
    while (is_space(s[i])) {
      if (++i == l)
	return output;
    }
    size_t j;
    for (j = i+1; (j < l) && !is_space(s[j]); j++) {
      (void)0;
    }
    output.push_back(s.substr(i, j-i));
    i = j;
  }
  return output;
}

const Job::Variable Job::no_variable;

const Job::Variable
Job::parse_line(const std::string &s)
{
  std::vector<std::string> data = split_string(s);
  // Format:
  //  variable_name min_value max_value [n_steps]
  // where
  //  variable_name is any string
  //  min_value, max_value are real
  //  n_steps is a non-negative integer, defaulting to 10
  if (data.size() == 0)
    return no_variable;
  if (data.size() == 1) {
    std::cerr << "Missing minimum value for variable " << data[0] << std::endl;
    return no_variable;
  }
  double min = atof(data[1].c_str());
  if (data.size() == 2) {
    std::cerr << "Missing maximum value for variable " << data[0] << std::endl;
    return no_variable;
  }
  double max = atof(data[2].c_str());
  if (data.size() > 4) {
    std::cerr << "Too many arguments for variable " << data[0] << std::endl;
    return no_variable;
  }
  tensor::index n = (data.size() == 3)? 10 : atoi(data[3].c_str());
  return Variable(data[0], min, max, n);
}

int
Job::parse_file(std::istream &s, std::vector<Variable> &data)
{
  std::string buffer;
  int line;
  for (line = 0; 1; line++) {
    const Variable v = parse_line(buffer);
    if (v.name().size() == 0)
      return line;
    else
      data.push_back(v);
  } while(1);
  return 0;
}

Job::Job(int argc, const char **argv)
{
  bool loaded = false;
  bool print_jobs = false;
  _this_job = 0;
  int i;
  for (i = 0; i < argc; i++) {
    if (strcmp(argv[i], "--job") == 0) {
      if (++i == argc) {
	std::cerr << "Missing argument after --job" << std::endl;
	abort();
      }
      _filename = std::string(argv[i]);
      std::ifstream s(argv[i]);
      int line = parse_file(s, _variables);
      if (line) {
	std::cerr << "Syntax error in line " << line << " of job file " << _filename
		  << std::endl;
	abort();
      }
      loaded = true;
    } else if (strcmp(argv[i], "--print-jobs")) { 
      print_jobs = true;
    } else if (strcmp(argv[i], "--this-job")) {
      if (++i == argc) {
	std::cerr << "Missing argument to --this-job" << std::endl;
	_this_job = atoi(argv[i]);
      }
    }
  }
  if (!loaded) {
    std::cerr << "Missing --job file option" << std::endl;
    abort();
  } else if (_variables.size() == 0) {
    std::cerr << "Job file " << _filename << " contained no varibles" << std::endl;
    abort();
  } else {
    tensor::index i = _this_job;
    _number_of_jobs = 1;
    for (std::vector<Variable>::iterator it = _variables.begin();
	 it != _variables.end();
	 it++) {
      tensor::index n = it->size();
      _number_of_jobs *= n;
      it->select(i % n);
      i = i / n;
    }
  }
  if (print_jobs) {
    std::cout << number_of_jobs();
    exit(0);
  }
}

void
Job::select_job(tensor::index which)
{
  if (which >= _number_of_jobs || which < 0) {
    std::cerr << "Cannot select job " << which << " out of "
	      << _number_of_jobs << " in Job file " << _filename
	      << std::endl;
    abort();
  } else {
    tensor::index i = _this_job = which;
    for (std::vector<Variable>::iterator it = _variables.begin();
	 it != _variables.end();
	 it++) {
      tensor::index n = it->size();
      it->select(i % n);
      i = i / n;
    }
  }
}

double
Job::get_value(const std::string &variable) const
{
  for (std::vector<Variable>::const_iterator it = _variables.begin();
       it != _variables.end();
       it++) {
    if (it->name() == variable) {
      return it->value();
    }
  }
  std::cerr << "Variable " << variable << " not found in job file "
	    << _filename << std::endl;
  abort();
}
