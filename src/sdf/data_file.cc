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

#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/file.h>
#include <fcntl.h>
#include <tensor/config.h>
#include <tensor/sdf.h>

using namespace sdf;

#ifdef TENSOR_BIGENDIAN
const enum DataFile::endianness DataFile::endian = BIG_ENDIAN_FILE;
#else
const enum DataFile::endianness DataFile::endian = LITTLE_ENDIAN_FILE;
#endif

/* Try to get lock. Return its file descriptor or -1 if failed.
 */
static int get_lock(char const *lockName, bool wait)
{
    int fd;
    do {
      mode_t m = umask(0);
      fd = open(lockName, O_RDWR|O_CREAT, 0666);
      umask(m);
      if( fd >= 0 && flock( fd, LOCK_EX | LOCK_NB ) < 0 )
	{
	  close(fd);
	  fd = -1;
	}
      if (fd > 0 || !wait)
	return fd;
      sleep(1);
    } while (1);
}

/* Release the lock obtained with tryGetLock( lockName ).
 */
static void giveup_lock(int fd, char const *lockName)
{
    if( fd < 0 )
        return;
    unlink(lockName);
    close(fd);
}


const size_t DataFile::var_name_size;

DataFile::DataFile(const std::string &a_filename, bool lock) :
  _filename(a_filename), _lock_filename(a_filename + ".lck"),
  _lock(lock? get_lock(_lock_filename.c_str(), true) : 0),
  _open(true)
{
}

DataFile::~DataFile()
{
  close();
}

void
DataFile::close()
{
  if (is_open()) {
    _open = false;
    if (is_locked()) {
      giveup_lock(_lock, _lock_filename.c_str());
    }
  }
}

const char *
DataFile::tag_to_name(size_t tag)
{
    static const char *names[] = {
	"RTensor", "CTensor", "Real MPS", "Complex MPS"
    };

    if (tag > 4 || tag < 0) {
	std::cerr << "Not a valid tag code, " << tag << " found in " << _filename;
    }
    return names[tag];
}
