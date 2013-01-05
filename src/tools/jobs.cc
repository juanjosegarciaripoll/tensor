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
bool is_separator(char c)
{
  return (c == ' ') || (c == '\t') || (c == '\n') || (c == ',') || (c == ';');
}

std::vector<std::string>
split_string(const std::string &s)
{
  std::vector<std::string> output;
  size_t i = 0, l = s.size();
  while (i != l) {
    while (is_separator(s[i])) {
      if (++i == l)
	return output;
    }
    size_t j;
    for (j = i+1; (j < l) && !is_separator(s[j]); j++) {
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
  //  variable_name separator min_value separator max_value [separator n_steps]
  // where
  //  variable_name is any string
  //  min_value, max_value are real
  //  n_steps is a non-negative integer, defaulting to 10
  //  separator is any number of spaces, tabs, newlines, commas or semicolons
  if (data.size() == 0)
    return no_variable;
  if (data.size() == 1) {
    std::cerr << "Missing minimum value for variable " << data[0] << std::endl;
    return no_variable;
  }
  double max, min = atof(data[1].c_str());
  tensor::index nsteps;
  if (data.size() == 2) {
    max = min;
    nsteps = 1;
  } else {
    max = atof(data[2].c_str());
    if (data.size() == 3) {
      nsteps = 10;
    } else if (data.size() > 4) {
      std::cerr << "Too many arguments for variable " << data[0] << std::endl;
      return no_variable;
    } else {
      nsteps = atoi(data[3].c_str());
    }
  }
  return Variable(data[0], min, max, nsteps);
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

Job::Job(int argc, const char **argv) :
  _filename("no file")
{
  bool loaded = false;
  bool print_jobs = false;
  _this_job = 0;
  int i;
  for (i = 0; i < argc; i++) {
    if (!strcmp(argv[i], "--job")) {
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
    } else if (!strcmp(argv[i], "--print-jobs")) { 
      print_jobs = true;
    } else if (!strcmp(argv[i], "--this-job")) {
      if (++i == argc) {
	std::cerr << "Missing argument to --this-job" << std::endl;
	abort();
      }
      _this_job = atoi(argv[i]);
    } else if (!strcmp(argv[i], "--variable")) {
      if (++i == argc) {
	std::cerr << "Missing argument after --variable" << std::endl;
	abort();
      }
      Variable v = parse_line(argv[i]);
      if (v.name().size() == 0) {
	std::cerr << "Syntax error parsing --variable argument:" << std::endl
		  << argv[i] << std::endl;
	abort();
      }
      _variables.push_back(v);
      loaded = true;
    } else if (!strcmp(argv[i], "--help")) {
      std::cout << "Arguments:\n"
	"--help\n"
	"\tShow this message and exit.\n"
	"--print-jobs\n"
	"\tShow number of jobs and exit.\n"
	"--this-job n\n"
	"\tChoose which job this is.\n"
	"--variable name,min[,max[,nsteps]]\n"
	"\tDefine variable and the range it runs, including number of steps.\n"
	"\t'min' and 'max' are floating point numbers; 'name' is a string\n"
	"\twithout spaces; 'nsteps' is a positive (>0) integer denoting how\n"
	"\tmany values in the interval [min,max] are used. There can be many\n"
	"\t--variable arguments, as a substitute for a job file. If both 'max'\n"
	"\tand 'nsteps' are missing, the variable takes a unique, constant\n"
	"\tvalue, 'min'.\n"
	"--job filename\n"
	"\tParse file, loading variable ranges from them using the syntax above,\n"
	"\tbut allowing spaces or tabs instead of commas as separators."
		<< std::endl;
      exit(0);
    }
  }
  if (!loaded) {
    std::cerr << "Missing --job or --variable options" << std::endl;
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


double
Job::get_value_with_default(const std::string &variable, double def) const
{
  for (std::vector<Variable>::const_iterator it = _variables.begin();
       it != _variables.end();
       it++) {
    if (it->name() == variable) {
      return it->value();
    }
  }
  return def;
}
