@echo off
:main
set _TENSOR_ERROR=no
set _TENSOR_ARCH=
set _TENSOR_CLEAN=
set _TENSOR_CONFIGURE=
set _TENSOR_BUILDDIR=
set _TENSOR_BUILD=
set _TENSOR_TEST=
set _TENSOR_THREADS=1
set _TENSOR_CMAKE_OPTIONS=-DCMAKE_TOOLCHAIN_FILE=%HOME%/src/vcpkg/scripts/buildsystems/vcpkg.cmake
set CMAKE_BUILD_TYPE=Release
call :setenv build-and-test.cmd
if "x%1" NEQ "x" call :process build-and-test.cmd %1
if "x%2" NEQ "x" call :process build-and-test.cmd %2
if "x%3" NEQ "x" call :process build-and-test.cmd %3
if "x%4" NEQ "x" call :process build-and-test.cmd %4
if "x%5" NEQ "x" call :process build-and-test.cmd %5
if "x%6" NEQ "x" call :process build-and-test.cmd %6
if "x%7" NEQ "x" call :process build-and-test.cmd %7
if "x%8" NEQ "x" call :process build-and-test.cmd %8
if "x%9" NEQ "x" call :process build-and-test.cmd %9
if "%_TENSOR_ERROR%" EQU "yes" goto :error

set _TENSOR_BUILDDIR="%cd%\build-%_TENSOR_ARCH%"
echo Architecture: %_TENSOR_ARCH%
echo Build directory: %_TENSOR_BUILD%
echo Error: %_TENSOR_ERROR%
if "%_TENSOR_CLEAN%" EQU "yes" call :clean build-and-test.cmd
if "%_TENSOR_CONFIGURE%" EQU "yes" call :configure build-and-test.cmd
if "%_TENSOR_BUILD%" EQU "yes" call :build build-and-test.cmd
if "%_TENSOR_TEST%" EQU "yes" call :test build-and-test.cmd
goto :eof

:process
if "%2" EQU "--help" goto :help
if "%2" EQU "--clean" (
  set _TENSOR_CLEAN=yes
  goto :eof
)
if "%2" EQU "--all" (
  set _TENSOR_CONFIGURE=yes
  set _TENSOR_BUILD=yes
  set _TENSOR_TEST=yes
  goto :eof
)
if "%2" EQU "--configure" (
  set _TENSOR_CONFIGURE=yes
  goto :eof
)
if "%2" EQU "--build" (
  set _TENSOR_BUILD=yes
  goto :eof
)
if "%2" EQU "--debug" (
  set CMAKE_BUILD_TYPE=Debug
  goto :eof
)
if "%2" EQU "--release" (
  set CMAKE_BUILD_TYPE=Release
  goto :eof
)
if "%2" EQU "--rel-with-deb-info" (
  set CMAKE_BUILD_TYPE=RelWithDebInfo
  goto :eof
)
if "%2" EQU "--arpack" (
  set _TENSOR_CMAKE_OPTIONS=%_TENSOR_CMAKE_OPTIONS% -DTENSOR_ARPACK=ON
  goto :eof
)
if "%2" EQU "--fftw" (
  set _TENSOR_CMAKE_OPTIONS=%_TENSOR_CMAKE_OPTIONS% -DTENSOR_FFTW=ON
  goto :eof
)
if "%2" EQU "--test" (
  set _TENSOR_TEST=yes
  goto :eof
)
if "%2" EQU "-j4" (
  set _TENSOR_THREADS=4
  goto :eof
)
if "%2" EQU "-j8" (
  set _TENSOR_THREADS=8
  goto :eof
)
goto :setarch

rem
rem %%%%%%%%%%%%%% CLEAN DIRECTORY %%%%%%%%%%%%%%%%%
rem
:clean
if exist %_TENSOR_BUILDDIR% ( rmdir /s %_TENSOR_BUILDDIR% )
goto :eof

rem
rem %%%%%%%%%%%%%% CONFIGURE %%%%%%%%%%%%%%%%%
rem
:configure
if exist %_TENSOR_BUILDDIR% goto :doconfigure
mkdir %_TENSOR_BUILDDIR%

:doconfigure
cd %_TENSOR_BUILDDIR%
cmake .. %_TENSOR_CMAKE_OPTIONS% -DCMAKE_BUILD_TYPE=%CMAKE_BUILD_TYPE%
goto :eof

rem
rem %%%%%%%%%%%%%% BUILD %%%%%%%%%%%%%%%%%
rem
:build
cd %_TENSOR_BUILDDIR%
cmake --build . --parallel %_TENSOR_THREADS%
goto :eof

rem
rem %%%%%%%%%%%%%% TEST %%%%%%%%%%%%%%%%%
rem
:test
cd %_TENSOR_BUILDDIR%/tests
ctest --output-on-failure -j %_TENSOR_THREADS%
goto :eof

rem
rem %%%%%%%%%%%%%% ARCHITECTURE CHOICE %%%%%%%%%%%%%%%%%
rem

:setarch
if "x%_TENSOR_ARCH%" NEQ "x" goto :archdefined

:setarchvs17
if x%VS2017BTROOT% EQU x goto :setarchvs22
if "%2" EQU "vs17bt32" (
   %VS2017BTROOT%\vcvarsall.bat x86
   goto :archok
)
if "%2" EQU "vs17bt64" (
   %VS2017BTROOT%\vcvarsall.bat amd64
   goto :archok
)

:setarchvs22
if x%VS2022COMMROOT% EQU x goto :noarch
if "%2" EQU "vs22comm32" (
   call %VS2022COMMOOT%\vcvarsall.bat x86
   goto :archok
)
if "%2" EQU "vs22comm64" (
   call %VS2022COMMROOT%\vcvarsall.bat amd64
   goto :archok
)

:noarch
echo Unknown option %2
set _TENSOR_ERROR=yes
goto :eof

:archdefined
echo Architecture already defined
set _TENSOR_ERROR=yes
goto :eof

:archok
@echo on
set _TENSOR_ARCH=%2
goto :eof

rem
rem %%%%%%%%%%%%%% FIND BACKENDS %%%%%%%%%%%%%%%%%
rem

:setenv
set VS2017BTROOT="%ProgramFiles(x86)%\Microsoft Visual Studio\2017\BuildTools\VC\Auxiliary\Build"
if exist "%VS2017BTROOT%" () else (set VS2017BTROOT=)
set VS2022COMMROOT="%ProgramFiles%\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build"
if exist "%VS2022COMMROOT%" () else (set VS2022COMMROOT=)
exit /b

rem
rem %%%%%%%%%%%%%% HELP %%%%%%%%%%%%%%%%%
rem

:help
echo Available backends
if "X%VS2017BTROOT" NEQ "X" (
   echo vs17bt32 - Visual Studio Build Tools 2017 x86
   echo vs17bt64 - Visual Studio Build Tools 2017 amd64
   )
if "X%VS2022COMMROOT" NEQ "X" (
   echo vs22comm32 - Visual Studio Community 2022 x86
   echo vs22comm64 - Visual Studio Community 2022 amd64
   )
exit /b

:error
echo "Error"
exit -1
