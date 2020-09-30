DESCRIPTION
===========

This is a C++ library for numerical arrays and tensor objects and operations
with them, designed to allow Matlab-style programming. Some of the key features:

* Fundamental data structures
  - Arrays of indices
  - Multidimensional arrays of double precision, real and complex numbers
  - Sparse matrices of real and complex numbers

* Operations
  - Algebraic operations between tensors, matrices and numbers.
  - Contraction among tensors (a generalization of matrix multiplication).
  - Tensor reshaping, index permutation, enlarging and contracting.

* Linear algebra
  - Solving systems of linear equations
  - Matrix exponentiation
  - Eigenvalue problems
  - Few eigenvalues of sparse problems using Arpack
  - Singular value decompositions

* Fast Fourier Transforms (to be done)

The Tensor library is discussed in the libtensor Google Group, currently found
here https://groups.google.com/forum/#!forum/libtensor Go to this forum for
help in configuring and installing the library.

Bug reports should be submitted using the GitHub Issues interface.
  

BUILD AND INSTALL
=================

Follow these steps to prepare, configure, build and install this library:

1) If you have checked out the library using a version control system (git) then
   several files have to be rebuilt before moving any further. Open a terminal
   and enter

      ./autogen.sh

   This will use the autotools (autoconf, automake, libtool) to rebuild several
   files, such as Makefile.in, configure, files in the m4 directory, etc.

2) Configure the library. This process detects existing software and chooses
   one or more options, such as building statically linked or shared libraries,
   using the Google Test library, etc. The process involves again a terminal
   and typing something like

      ./configure --prefix=$HOME LIBS="..." CXXFLAGS="..."

   Here we are using --prefix=$HOME to tell the configuration program that the
   libraries are going to be installed in our home directory, under $HOME/lib
   and $HOME/include, and we are passing additional options ("...") such as a
   list of libraries that are needed (LIBS) and flags for the C++ compiler
   (See below)

3) Build, optionally check and install the library

      make
      make check # optional
      make install


PLATFORM SPECIFIC DETAILS
=========================

LINUX / *BSD / etc
------------------

   You will need some version of the LAPACK and BLAS libraries installed.
   Typically, with Debian systems there are the Atlas libraries. To tell
   the tensor library that you are going to use them pass the option

       LIBS="-llapack -lcblas -latlas"

   to the "configure" program listed above.

MAC OS X
--------

   The OS X operating system is shipped with an optimized version of the Atlas
   libraries that can be used by passing the option

       LIBS="-framework veclib"

   to "configure"


OPTIONAL COMPONENTS
===================

Testing
-------

   To test the library you will need the Google Test framework, which is
   available at

       http://code.google.com/p/googletest/

Documentation
-------------

   Documentation is built using the Doxygen package, which is available in
   most Linux-type software distributions

       http://www.stack.nl/~dimitri/doxygen/

   The documentation itself is built using the command "make doxygen-doc"
   after "make" and before "make install"