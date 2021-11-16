Building and installing the tensor library
==========================================

The library uses CMake for generating the build files.
In theory, you can build everything with the following sequence of commands

> cmake -S ${SourceDir} -B ${BinaryDir}
> make -C ${BinDir} install

This will build the library using make as build tool and install it to
a default directory (`/usr/local` under Unix). ${BinaryDir} is the build
directory and can be deleted after the installation.

CMake can be configured by setting user variables with `-D VAR=<value>`.
The most useful variables here are

* CMAKE_CXX_FLAGS_RELEASE for specifying flags to the compiler for a release build
* CMAKE_SHARED_LINKER_FLAGS for specifying flags to the linker for a release build
  of a shared library
* INSTALL_PREFIX to install the artefacts to a non-standard location
  (/usr/local on Unix, C:\Program Files\<project_name> or so on Windows)
* BUILD_SHARED_LIBS. Setting this to ON builds libraries as shared libraries (.so/.dll),
  OFF builds static libraries.
* TENSOR_ARPACK, TENSOR_FFTW, TENSOR_TEST. Setting any of these to OFF deactivates the
  corresponding tensor code for large eigenvalue problems, FFT or the unit tests.

The most likely problems that you will encounter is the build of the dependencies.
Dependencies of the tensor library are:

* CMake version 3.13 or newer
* Blas; currently OpenBLAS and MKL are supported out of the box
  everything else requires manual fiddling.
* Lapack; MKL brings its own library here
* (optional) Fftw for FFT support
* (optional) Arpack-ng for support with large-scale eigenvalue problems
* (optional) GoogleTest to run the unit tests

Unfortunately, not all dependencies export their metadata with CMake in all
setups. For example, Debian packages come with pkgconfig files, but not CMake,
and the OpenBlas package comes with special directories, which makes it
difficult to utilize these libs by default. In general, especially Blas and
\*pack libraries are rather painful; for example the MKL seems to change the
whole directory layout every now and again, which prevents the use of static recipes.

For simple setups, we recommend using vcpkg, which makes it rather easy to get
the library up and running. For advanced setups, or for highest performance,
you may need to deal with the dependencies one by one, as described later
in the document.


vcpkg without Arpack
--------------------

Arpack is not yet available in vcpkg, so we need to deal with it separately.
For simplicity, we start with the recipe without Arpack.

Conceptually, when using vcpkg, the following happens

* We ask vcpkg to install the requirements. 
  vcpkg downloads and builds them internally with all transitive dependencies.
* When building the tensor library, we pull in vcpkg in the CMake step.
  On one hand, all searches are then redirected to preferably find the vcpkg-built
  libraries. On the other hand, vcpkg may hook into the build for some additional
  magic.

Note that vcpkg works such that the result of your build is self-contained without
external dependencies. Under Linux, this means that all vcpkg artefacts are static
libraries that are directly linked into the tensor library. Under Windows, vcpkg
hooks into the build to copy all dependent DLLs into the output / install folder
(effectively, Windows requires all dependencies to be in the same path as the tensor
library).

The steps in detail:

* Clone the vcpkg repository:  
  `cd somedir; git clone https://github.com/microsoft/vcpkg; cd vcpkg`  
  The directory you are now in will be called ${VCPKG} in the following.
* (optional) reset the repository to a stable state that was used for this Readme  
  `git checkout 2021.05.12`  
  This walkthrough should work with the latest repository as well, though.
* Initialize vcpkg  
  `./bootstrap-vcpkg.sh`  
  This downloads and builds the executable, and installs some well-defined tools.
* Install all required packages  
  `./vcpkg install openblas fftw3 lapack gtest`  
  By default, everything is compiled from scratch; go get some coffee.

When this is done, you can compile the tensor library.

> cmake -D BUILD_SHARED_LIBS=ON  
>       -D CMAKE_BUILD_TYPE=Release
>       -S ${SourceDir} -B ${BinaryDir}  
>       -D TENSOR_ARPACK=OFF  
>       -D CMAKE_TOOLCHAIN_FILE=${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake  
> make -C ${BinaryDir} install

The first three lines instruct CMake to build an optimized shared library,
the fourth line disables the use of Arpack code, and the last line includes the
vcpkg magic. Note that the toolchain path should be absolute. More definitions can
be added, of course.


Building with Arpack
--------------------

If you want the Arpack functionality, you need to also build Arpack. Unfortunately,
this library has no vcpkg port, so you need to build it yourself. The current stable
version also has some smaller pecularities.

* Download Arpack-ng  
  `git clone https://github.com/opencollab/arpack-ng`  
  `git checkout 3.8.0`  
  The latter command just uses a stable release for safety.
* Go to the source directory and build Arpack  
  `cmake -D BUILD_SHARED_LIBS=ON -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=${ArpackInstall} -S . -B build`  
  `make install`  
  Afterwards the directory "build" can be deleted. The ${ArpackInstall} prefix defaults to something
  like /usr/local on Unix if not given.
* Build the tensor library. The flags are slightly different  
  `cmake -D BUILD_SHARED_LIBS=ON
         -S ${SourceDir} -B ${BinaryDir}  
         -D CMAKE_BUILD_TYPE=Release
         -D CMAKE_TOOLCHAIN_FILE=${VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake  
         -D CMAKE_PREFIX_PATH=${ArpackInstall}  
         -D CMAKE_SHARED_LINKER_FLAGS_RELEASE=-L${ArpackInstall}/lib
         -D CMAKE_EXE_LINKER_FLAGS_RELEASE=-L${ArpackInstall}/lib`  
  Ok, what happened here?  
  The script that Arpack-ng uses for telling other CMake projects the flags is broken.
  Among other things, it does not set the directory where to search for the library,
  hence we need to set it ourselves (can be skipped if arpack-ng is installed e.g. to
  /usr/local, this is a hardcoded CMake default), and we need to set it for the linking
  of shared libraries _and_ exectuables (test runners). The prefix path adds the installation
  path of arpack-ng to the path below which CMake searches for the modules.
* Now you can compile. Note that, unless you disable tests, you need to include the
  directory with the arpack-ng library into your runtime search path,
  e.g., by setting LD\_LIBRARY\_PATH to ${ArpackInstall}/lib.
  `export LD_LIBRARY_PATH=${ArpackInstall}/lib  
  make -C ${BinaryDir}`
