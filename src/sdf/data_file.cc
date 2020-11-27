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

#if !defined(_MSC_VER) && !defined(__MINGW32__)
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/file.h>
#include <fcntl.h>
#include <unistd.h>
#endif
#include <tensor/config.h>
#include <tensor/sdf.h>
#include <errno.h>

namespace sdf {

#ifdef TENSOR_BIGENDIAN
const enum DataFile::endianness DataFile::endian = BIG_ENDIAN_FILE;
#else
const enum DataFile::endianness DataFile::endian = LITTLE_ENDIAN_FILE;
#endif
const unsigned int DataFile::var_name_size = 64;

/* Try to get lock. Return its file descriptor or -1 if failed.
 */
static int get_lock(char const *lockName, bool wait) {
#if defined(_MSC_VER) || defined(__MINGW32__)
  return 1;
#else
  do {
    mode_t m = umask(0);
    int fd = open(lockName, O_RDWR | O_CREAT, 0666);
    umask(m);
    if (fd >= 0 && flock(fd, LOCK_EX | LOCK_NB) < 0) {
      if (errno == ENOTSUP) {
        std::cout << "Locking not supported. We assume you know what you are "
                     "doing.\n";
        return fd;
      }
      close(fd);
      fd = -1;
    }
    if (fd > 0 || !wait) return fd;
    sleep(1);
  } while (1);
#endif
}

/* Release the lock obtained with tryGetLock( lockName ).
 */
static void giveup_lock(int fd, char const *lockName) {
#if !defined(_MSC_VER) && !defined(__MINGW32__)
  if (fd < 0) return;
  unlink(lockName);
  close(fd);
#endif
}

DataFile::DataFile(const std::string &a_filename, int flags)
    : _flags(flags),
      _actual_filename(a_filename),
      _filename(_actual_filename),
      _lock_filename(_actual_filename + ".lck"),
      _lock((flags == SDF_SHARED) ? get_lock(_lock_filename.c_str(), true) : 0),
      _open(true) {
  switch (flags) {
    case SDF_OVERWRITE:
      if (file_exists(_actual_filename) && !delete_file(_actual_filename)) {
        std::cerr << "Cannot overwrite file " << _actual_filename << std::endl;
        abort();
      }
      break;
    case SDF_SHARED:
      break;
    case SDF_PARANOID:
      _actual_filename = _actual_filename + ".tmp";
      delete_file(_actual_filename);
      break;
    default:
      std::cerr << "Unrecognized DataFile mode " << flags << std::endl;
      abort();
  }
}

DataFile::~DataFile() { close(); }

void DataFile::close() {
  if (is_open()) {
    _open = false;
    switch (_flags) {
      case SDF_SHARED:
        if (is_locked()) {
          giveup_lock(_lock, _lock_filename.c_str());
        }
        break;
      case SDF_PARANOID:
        if (file_exists(_actual_filename))
          rename_file(_actual_filename, _filename);
    }
  }
}

const char *DataFile::tag_to_name(size_t tag) {
  static const char *names[] = {"RTensor", "CTensor", "Real MPS",
                                "Complex MPS"};

  if (tag > 4 /* || tag < 0 */) {
    std::cerr << "Not a valid tag code, " << tag << " found in " << _filename;
  }
  return names[tag];
}

}  // namespace sdf
