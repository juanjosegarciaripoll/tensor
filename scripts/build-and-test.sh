#!/bin/sh
do_test=yes
do_profile=yes
do_delete=yes
os=`uname -o`
threads=10
sourcedir=`pwd`
builddir="$sourcedir/build-$os"
logfile="$builddir/log"
OPENBLAS_NUM_THREADS=4
export OPENBLAS_NUM_THREADS
if [ -f /etc/os-release ]; then
   os=`(. /etc/os-release; echo $NAME)`
fi
for arg in $*; do
    case $arg in
        --no-test) do_test=no;;
        --no-profile) do_profile=no;;
        --continue) do_delete=no;;
        --debug) set -x;;
    esac
done
if test -n `which ninja`; then
    generator="Ninja"
else
    generator="Unix Makefiles"
fi
if [ $do_delete = yes ]; then
    if [ -d "$builddir" ]; then
        rm -rf "$builddir"
    fi
fi
#
# Configure
#
test -d "$builddir" || mkdir "$builddir"
CMAKE_FLAGS="-DTENSOR_ARPACK=ON -DTENSOR_FFTW=ON -DTENSOR_CLANG_TIDY=ON -DTENSOR_OPTIMIZED_BUILD=ON"
cmake -H"$sourcedir" -B"$builddir" $CMAKE_FLAGS -G "$generator" 2>&1 | tee -a "$logfile"
if [ $? -ne 0 ]; then
    echo CMake configuration failed
    exit -1
fi
#
# Build
#
cmake --build "$builddir" --config Release -j $threads -- 2>&1 | tee -a "$logfile"
if [ $? -ne 0 ]; then
    echo CMake build failed
    exit -1
fi
#
# Run profile
#
if [ $do_profile = yes ]; then
    "$builddir/profile/profile" "$sourcedir/profile/benchmark_$os.json" 2>&1 | tee -a "$logfile"
    if [ $? -ne 0 ]; then
        echo CMake profile failed
        exit -1
    fi
fi
#
# Run tests
#
if [ $do_test = yes ]; then
    cd "$builddir"/tests && ctest -j $threads | tee -a "$logfile"
    if [ $? -ne 0 ]; then
        echo CMake profile failed
        exit -1
    fi
fi
