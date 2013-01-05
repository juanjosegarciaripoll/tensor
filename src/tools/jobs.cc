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
Job::parse_file(std::istream &s, var_list &data)
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
  filename_("no file")
{
  bool loaded = false;
  bool print_jobs = false;
  bool first_job_found = false;
  bool last_job_found = false;
  bool job_blocks_found = false;
  bool this_job_found = false;
  tensor::index blocks;
  tensor::index current_job;
  int i;
  for (i = 0; i < argc; i++) {
    if (!strcmp(argv[i], "--job")) {
      if (++i == argc) {
	std::cerr << "Missing argument after --job" << std::endl;
	abort();
      }
      filename_ = std::string(argv[i]);
      std::ifstream s(argv[i]);
      int line = parse_file(s, variables_);
      if (line) {
	std::cerr << "Syntax error in line " << line << " of job file " << filename_
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
      current_job = atoi(argv[i]);
    } else if (!strcmp(argv[i], "--job-blocks")) {
      if (++i == argc) {
	std::cerr << "Missing argument to --job-blocks" << std::endl;
	abort();
      }
      blocks = atoi(argv[i]);
      job_blocks_found = true;
    } else if (!strcmp(argv[i], "--first-job")) {
      if (this_job_found) {
        std::cerr << "Cannot use --first-job and --this-job simultaneously." << std::endl;
        abort();
      }
      if (++i == argc) {
	std::cerr << "Missing argument to --first-job" << std::endl;
	abort();
      }
      first_job_ = atoi(argv[i]);
      first_job_found = true;
    } else if (!strcmp(argv[i], "--last-job")) {
      if (!first_job_found) {
        std::cerr << "--last-job without preceding --first-job" << std::endl;
        abort();
      }
      if (++i == argc) {
	std::cerr << "Missing argument to --last-job" << std::endl;
	abort();
      }
      last_job_ = atoi(argv[i]);
      last_job_found = true;
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
      variables_.push_back(v);
      loaded = true;
    } else if (!strcmp(argv[i], "--help")) {
      std::cout << "Arguments:\n"
	"--help\n"
	"\tShow this message and exit.\n"
	"--print-jobs\n"
	"\tShow number of jobs and exit.\n"
        "--first-job / --last-job n\n"
        "\tRun this program completing the selected interval of jobs.\n"
	"--this-job n\n"
	"\tSelect to run one job or one job block.\n"
        "--jobs-blocks n\n"
        "\tSplit the number of blocks into 'n' blocks and run the jobs in the\n"
        "\tblock selected by --this-job.\n"
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
  number_of_jobs_ = compute_number_of_jobs();
  if (print_jobs) {
    std::cout << number_of_jobs_;
    exit(0);
  }
  if (job_blocks_found) {
    /*
     * We split the number of jobs into sets and we are going to run the
     * set indicated by --this-job (which defaults to 0)
     */
    if (first_job_found || last_job_found) {
      std::cerr << "The options --job-blocks and --first/last-job cannot be used together."
                << std::endl;
      abort();
    }
    if (blocks <= 0) {
      std::cerr << "The number of job blocks cannot be zero or negative.";
      abort();
    }
    tensor::index delta = std::max(number_of_jobs_ / blocks, (tensor::index)1);
    tensor::index actual_blocks = number_of_jobs_ / delta;
    if (current_job >= actual_blocks) {
      std::cerr << "warning: the value of --this-job exceeds the value of --job-blocks."
                << std::endl;
      abort();
    }
    first_job_ = current_job = current_job * delta;
    last_job_ = std::min(number_of_jobs_ - 1, first_job_ + delta - 1);
  } else if (first_job_found) {
    /*
     * --first-job and --last-job combined
     */
    if (first_job_ < 0) {
      std::cerr << "--first-job is negative" << std::endl;
      abort();
    } else if (first_job_ >= number_of_jobs_) {
      std::cerr << "--first-job exceeds the number of jobs" << std::endl;
      abort();
    }
    if (last_job_found) {
      if (last_job_ >= number_of_jobs_) {
        std::cerr << "warning: --last-job exceeds the number of jobs" << std::endl;
        last_job_ = std::max<tensor::index>(0, number_of_jobs_ - 1);
      }
      if (last_job_ < first_job_) {
        std::cerr << "--last-job is smaller than --first-job" << std::endl;
        abort();
      }
    }
    current_job = first_job_;
  } else if (first_job_found) {
    /*
     * Only --this-job, which allows us to do only one thing
     */
    first_job_ = last_job_ = current_job;
  } else {
    /*
     * No options: we run over all jobs
     */
    first_job_ = current_job = 0;
    last_job_ = number_of_jobs_ - 1;
  }
  select_job(current_job);
}

tensor::index
Job::compute_number_of_jobs() const
{
  tensor::index n = 1;
  for (var_list::const_iterator it = variables_.begin();
       it != variables_.end();
       it++) {
    n *= it->size();
  }
  return n;
}

void
Job::select_job(tensor::index which)
{
  assert(which >= 0);
  tensor::index i = this_job_ = which;
  if (which <= number_of_jobs_) {
    for (var_list::iterator it = variables_.begin();
         it != variables_.end();
         it++) {
      tensor::index n = it->size();
      it->select(i % n);
      i = i / n;
    }
  }
}

void
Job::operator++()
{
  select_job(this_job_ + 1);
}

bool
Job::to_do()
{
  return (this_job_ >= 0) &&
    (this_job_ < number_of_jobs_) &&
    (this_job_ >= first_job_) &&
    (this_job_ <= last_job_);
}

const Job::Variable *
Job::find_variable(const std::string &name) const
{
  for (var_list::const_iterator it = variables_.begin(); it != variables_.end(); it++) {
    if (it->name() == name) {
      return &(*it);
    }
  }
  return NULL;
}

double
Job::get_value(const std::string &name) const
{
  const Variable *which = find_variable(name);
  if (!which) {
    std::cerr << "Variable " << name << " not found in job file "
              << filename_ << std::endl;
    abort();
  }
  return which->value();
}


double
Job::get_value_with_default(const std::string &name, double def) const
{
  const Variable *which = find_variable(name);
  return which? which->value() : def;
}
