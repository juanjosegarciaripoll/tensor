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

#include <csignal>
#include <cstdlib>
#include <cstdio>
#include <tensor/config.h>
#include <tensor/tools.h>

#ifdef HAVE_DLADDR
#ifdef HAVE_DLFCN_H
#include <dlfcn.h>
#endif
#endif

#if defined(HAVE_BACKTRACE) || defined(HAVE_BACKTRACE_SYMBOLS)
#include <execinfo.h>
#endif

#if !defined(HAVE_BACKTRACE) && defined(HAVE___BUILTIN_RETURN_ADDRESS) && \
    defined(HAVE_BACKTRACE_SYMBOLS)
#define HAVE_BACKTRACE
static int backtrace(void **buffer, int n) {
  int nframes = (n > 32) ? 32 : n;
  switch (nframes) {
    case 32:
      buffer[31] = __builtin_return_address(31);
    case 31:
      buffer[30] = __builtin_return_address(30);
    case 30:
      buffer[29] = __builtin_return_address(29);
    case 29:
      buffer[28] = __builtin_return_address(28);
    case 28:
      buffer[27] = __builtin_return_address(27);
    case 27:
      buffer[26] = __builtin_return_address(26);
    case 26:
      buffer[25] = __builtin_return_address(25);
    case 25:
      buffer[24] = __builtin_return_address(24);
    case 24:
      buffer[23] = __builtin_return_address(23);
    case 23:
      buffer[22] = __builtin_return_address(22);
    case 22:
      buffer[21] = __builtin_return_address(21);
    case 21:
      buffer[20] = __builtin_return_address(20);
    case 20:
      buffer[19] = __builtin_return_address(19);
    case 19:
      buffer[18] = __builtin_return_address(18);
    case 18:
      buffer[17] = __builtin_return_address(17);
    case 17:
      buffer[16] = __builtin_return_address(16);
    case 16:
      buffer[15] = __builtin_return_address(15);
    case 15:
      buffer[14] = __builtin_return_address(14);
    case 14:
      buffer[13] = __builtin_return_address(13);
    case 13:
      buffer[12] = __builtin_return_address(12);
    case 12:
      buffer[11] = __builtin_return_address(11);
    case 11:
      buffer[10] = __builtin_return_address(10);
    case 10:
      buffer[9] = __builtin_return_address(9);
    case 9:
      buffer[8] = __builtin_return_address(8);
    case 8:
      buffer[7] = __builtin_return_address(7);
    case 7:
      buffer[6] = __builtin_return_address(6);
    case 6:
      buffer[5] = __builtin_return_address(5);
    case 5:
      buffer[4] = __builtin_return_address(4);
    case 4:
      buffer[3] = __builtin_return_address(3);
    case 3:
      buffer[2] = __builtin_return_address(2);
    case 2:
      buffer[1] = __builtin_return_address(1);
    case 1:
      buffer[0] = __builtin_return_address(0);
  }
  return nframes;
}
#endif

#if !defined(HAVE_BACKTRACE_SYMBOLS)
#if defined(HAVE_BACKTRACE) && defined(HAVE_DLADDR)
#define HAVE_BACKTRACE_SYMBOLS
#define BACKTRACE_SYMBOLS_SIMPLE
static char **backtrace_symbols(void **buffer, int nframes) {
  Dl_info data[1];
  int i;
  char **strings = malloc(nframes * sizeof(char *));
  for (i = 0; i < nframes; i++) {
    if (dladdr(buffer[i], data)) {
      strings[i] = data->dli_sname;
    } else {
      strings[i] = "unknown";
    }
  }
  return strings;
}
#endif /* HAVE_BACKTRACE && HAVE_DLADDR */
#endif /* !HAVE_BACKTRACE_SYMBOLS */

#ifdef HAVE_BACKTRACE_SYMBOLS
static void dump_backtrace() {
  {
    void *pointers[32];
    int nframes = backtrace(pointers, 32);
    char **names = backtrace_symbols(pointers, nframes);
    int i;
    fprintf(stderr, "\n;;; ECL C Backtrace\n");
    for (i = 0; i < nframes; i++) {
#ifdef BACKTRACE_SYMBOLS_SIMPLE
      fprintf(stderr, ";;; %4d %s (%p) \n", i, names[i], pointers[i]);
#else
      fprintf(stderr, ";;; %s\n", names[i]);
#endif
    }
    fflush(stderr);
    free(names);
  }
}
#else
static void dump_backtrace() {}
#endif

#undef abort

static void tensor_abort(int /*signal*/) {
  dump_backtrace();
  exit(-1);
}

void tensor::tensor_abort_handler() { signal(SIGABRT, tensor_abort); }
