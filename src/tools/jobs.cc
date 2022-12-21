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

#include <algorithm>
#include <functional>
#include <numeric>
#include <tensor/jobs.h>
#include <fstream>
#include <cstring>

using namespace jobs;

static inline bool is_separator(char c) {
  return (c == ' ') || (c == '\t') || (c == '\n') || (c == ',') || (c == ';');
}

std::vector<std::string> split_string(const std::string &s) {
  std::vector<std::string> output;
  size_t i = 0, l = s.size();
  while (i != l) {
    while (is_separator(s[i])) {
      if (++i == l) return output;
    }
    size_t j = i + 1;
    for (; (j < l) && !is_separator(s[j]); ++j) {
      (void)0;
    }
    output.push_back(s.substr(i, j - i));
    i = j;
  }
  return output;
}

const Job::Variable Job::no_variable;

const Job::Variable Job::parse_line(const std::string &s) {
  std::vector<std::string> data = split_string(s);
  // Format:
  //  variable_name separator min_value separator max_value [separator n_steps]
  // where
  //  variable_name is any string
  //  min_value, max_value are real
  //  n_steps is a non-negative integer, defaulting to 10
  //  separator is any number of spaces, tabs, newlines, commas or semicolons
  if (data.size() == 0) return no_variable;
  if (data.size() == 1) {
    std::cerr << "Missing minimum value for variable " << data[0] << std::endl;
    return no_variable;
  }
  double min = atof(data[1].c_str());
  double max = min;
  index nsteps = 0;
  if (data.size() == 2) {
    nsteps = 1;
  } else {
    max = atof(data[2].c_str());
    if (data.size() == 3) {
      nsteps = Job::default_steps;
    } else if (data.size() > 4) {
      std::cerr << "Too many arguments for variable " << data[0] << std::endl;
      return no_variable;
    } else {
      nsteps = atoi(data[3].c_str());
    }
  }
  return Variable(data[0], min, max, nsteps);
}

int Job::parse_file(std::istream & /*s*/, var_list &data) {
  std::string buffer;
  int line = 0;
  while (true) {
    const Variable v = parse_line(buffer);
    if (v.name().size() == 0)
      return line;
    else
      data.push_back(v);
    ++line;
  }
  return 0;
}

static void print_job_help_and_exit() {
  std::cout
      << "Arguments:\n"
         "--help\n"
         "\tShow this message and exit.\n"
         "--print-jobs\n"
         "\tShow number of jobs and exit.\n"
         "--first-job / --last-job n\n"
         "\tRun this program completing the selected interval of jobs.\n"
         "--this-job n\n"
         "\tSelect to run one job or one job block.\n"
         "--jobs-blocks n\n"
         "\tSplit the number of blocks into 'n' blocks and run the jobs in "
         "the\n"
         "\tblock selected by --this-job.\n"
         "--variable name,min[,max[,nsteps]]\n"
         "\tDefine variable and the range it runs, including number of "
         "steps.\n"
         "\t'min' and 'max' are floating point numbers; 'name' is a "
         "string\n"
         "\twithout spaces; 'nsteps' is a positive (>0) integer denoting "
         "how\n"
         "\tmany values in the interval [min,max] are used. There can be "
         "many\n"
         "\t--variable arguments, as a substitute for a job file. If both "
         "'max'\n"
         "\tand 'nsteps' are missing, the variable takes a unique, "
         "constant\n"
         "\tvalue, 'min'.\n"
         "--job filename\n"
         "\tParse file, loading variable ranges from them using the syntax "
         "above,\n"
         "\tbut allowing spaces or tabs instead of commas as separators.";
  exit(0);
}

Job::Job(int argc, const char **argv) : filename_("no file") {
  bool print_jobs = false;
  bool first_job_found = false;
  bool last_job_found = false;
  bool job_blocks_found = false;
  bool this_job_found = false;
  index blocks{0};
  index current_job_number{0};

  const auto argv_end = argv + argc;
  auto arguments_left = [&]() { return argv != argv_end; };
  auto pop_argument = [&]() {
    tensor_assert(arguments_left());
    const char *output = *argv;
    ++argv;
    return output;
  };

  auto get_option_argument = [&](const char *option) {
    if (!arguments_left()) {
      std::cerr << "Missing argument after " << option << 'n';
      abort();
    }
    return pop_argument();
  };

  auto parse_job_filename = [&](const char *argument) {
    filename_ = std::string(argument);
    std::ifstream s(argument);
    int line = parse_file(s, variables_);
    if (line) {
      std::cerr << "Syntax error in line " << line << " of job file "
                << filename_ << '\n';
      abort();
    }
  };

  auto parse_integer = atoi;

  auto parse_variable = [&](const char *argument) {
    Variable v = parse_line(argument);
    if (v.name().size() == 0) {
      std::cerr << "Syntax error parsing --variable argument:\n"
                << argument << '\n';
      abort();
    }
    return v;
  };

  while (arguments_left()) {
    const char *option = pop_argument();
    if (!strcmp(option, "--job")) {
      parse_job_filename(get_option_argument("--job"));
    } else if (!strcmp(option, "--print-jobs")) {
      print_jobs = true;
    } else if (!strcmp(option, "--this-job")) {
      current_job_number = parse_integer(get_option_argument("--this-job"));
      this_job_found = true;
    } else if (!strcmp(option, "--job-blocks")) {
      blocks = parse_integer(get_option_argument("--job-blocks"));
      job_blocks_found = true;
    } else if (!strcmp(option, "--first-job")) {
      first_job_ = parse_integer(get_option_argument("--first-job"));
      first_job_found = true;
    } else if (!strcmp(option, "--last-job")) {
      if (!first_job_found) {
        std::cerr << "--last-job without preceding --first-job\n";
        abort();
      }
      last_job_ = parse_integer(get_option_argument("--last-job"));
      last_job_found = true;
    } else if (!strcmp(option, "--variable")) {
      variables_.push_back(parse_variable(get_option_argument("--variable")));
    } else if (!strcmp(option, "--help")) {
      print_job_help_and_exit();
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
      std::cerr << "The options --job-blocks and --first/last-job cannot be "
                   "used together."
                << std::endl;
      abort();
    }
    if (blocks <= 0) {
      std::cerr << "The number of job blocks cannot be zero or negative.";
      abort();
    }
    index delta = (number_of_jobs_ + blocks - 1) / blocks;
    first_job_ = current_job_number = std::min(current_job_number * delta, number_of_jobs_);
    last_job_ = std::min(number_of_jobs_ - 1, first_job_ + delta - 1);
  } else if (first_job_found) {
    /*
     * --first-job and --last-job combined
     */
    if (first_job_ < 0) {
      std::cerr << "--first-job is negative\n";
      abort();
    } else if (first_job_ >= number_of_jobs_) {
      std::cerr << "--first-job exceeds the number of jobs\n";
      abort();
    }
    if (last_job_found) {
      if (last_job_ >= number_of_jobs_) {
        std::cerr << "warning: --last-job exceeds the number of jobs"
                  << std::endl;
        last_job_ = std::max<index>(0, number_of_jobs_ - 1);
      }
      if (last_job_ < first_job_) {
        std::cerr << "--last-job is smaller than --first-job\n";
        abort();
      }
    }
    current_job_number = first_job_;
  } else if (this_job_found) {
    /*
     * Only --this-job, which allows us to do only one thing
     */
    first_job_ = last_job_ = current_job_number;
  } else {
    /*
     * No options: we run over all jobs
     */
    first_job_ = current_job_number = 0;
    last_job_ = number_of_jobs_ - 1;
  }
  select_job(current_job_number);
}

tensor::index Job::compute_number_of_jobs() const {
  return std::accumulate(variables_.cbegin(), variables_.cend(), index(1),
                         [](index x, auto &v) { return x * v.size(); });
}

void Job::select_job(index which) {
  tensor_assert(which >= 0);
  index i = this_job_ = which;
  if (which <= number_of_jobs_) {
    for (auto &variable : variables_) {
      index n = variable.size();
      variable.select(i % n);
      i = i / n;
    }
  }
}

void Job::operator++() { select_job(this_job_ + 1); }

bool Job::to_do() const {
  return (this_job_ >= 0) && (this_job_ < number_of_jobs_) &&
         (this_job_ >= first_job_) && (this_job_ <= last_job_);
}

const Job::Variable *Job::find_variable(const std::string &name) const {
  auto pos = std::find_if(variables_.cbegin(), variables_.cend(),
                          [&](const auto &v) { return v.name() == name; });
  if (pos == variables_.cend()) {
    return nullptr;
  } else {
    return &(*pos);
  }
}

double Job::get_value(const std::string &name) const {
  const Variable *which = find_variable(name);
  if (!which) {
    std::cerr << "Variable " << name << " not found in job file " << filename_
              << std::endl;
    abort();
  }
  return which->value();
}

double Job::get_value_with_default(const std::string &name, double def) const {
  const Variable *which = find_variable(name);
  return which ? which->value() : def;
}
