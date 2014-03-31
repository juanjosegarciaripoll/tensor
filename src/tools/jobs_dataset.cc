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

#include <string>
#include <memory>
#include <tensor/jobs.h>
#include <tensor/sdf.h>

namespace jobs {

  template <typename T>
  std::string NumberToString ( T Number )
  {
    /*
     * Input:
     *	T = a number
     * Output:
     *	a string with a representation of the number
     * Remarks:
     *	There is no way to control the length of this string or
     *	its accuracy.
     */
    std::ostringstream ss;
    ss << Number;
    return ss.str();
  }

  static std::string
  dataset_name(const std::string &base, int jobid)
  {
    return base + "/" + NumberToString(jobid);
  }

  Job::dataset
  Job::open_dataset(const std::string &filename) const
  {
    if (sdf::file_exists(filename)) {
      if (!sdf::isdir(filename)) {
	std::cerr << "Cannot create Job dataset on existing file\n";
	abort();
      }
    } else {
      if (!sdf::mkdir(filename)) {
	std::cerr << "Cannot create Job dataset\n";
	abort();
      }
    }
    std::string dataset_record_name = dataset_name(filename, current_job());
    sdf::OutDataFile *output;
    if (sdf::file_exists(dataset_record_name))
      output = NULL;
    else
      output = new sdf::OutDataFile(dataset_record_name,
				    sdf::DataFile::SDF_PARANOID);
    return dataset(output);
  }

  bool
  Job::dataset_record_exists(const std::string &filename) const
  {
    return sdf::file_exists(dataset_name(filename, current_job()));
  }

}
