option(TENSOR_DEFAULT_WARNINGS "Add known default compiler warnings" ON)
option(TENSOR_OPTIMIZED_BUILD "Add well known optimization arguments" ON)
option(TENSOR_CLANG_TIDY "Enable running clang-tidy if found" OFF)
option(TENSOR_CPPCHECK "Enable running cppcheck if found" OFF)
option(WARNINGS_AS_ERRORS "Compilation and analysis warnings become errors" OFF)
option(TENSOR_USE_PCH "Use precompiled headers" ON)
option(TENSOR_ADD_SANITIZERS "Compile and link with address sanitizers if in Debug mode" OFF)
option(TENSOR_COVERAGE "Add code coverage options to the library" OFF)

# The variables are not for ordinary users, so we hide them
mark_as_advanced(TENSOR_DEFAULT_WARNINGS
                 TENSOR_OPTIMIZED_BUILD
                 TENSOR_CLANG_TIDY
                 WARNINGS_AS_ERRORS
                 TENSOR_ADD_SANITIZERS
                 TENSOR_COVERAGE)

function(make_tensor_options)
    add_library(tensor_options INTERFACE)
    if (TENSOR_OPTIMIZED_BUILD)
        tensor_add_optimizations()
        if(TENSOR_COVERAGE)
            message(STATUS "Code coverage options disabled in release mode")
            set(TENSOR_COVERAGE OFF)
        endif()
    endif()

    if (TENSOR_DEFAULT_WARNINGS)
        tensor_add_warnings()
    endif()

    if (TENSOR_CLANG_TIDY)
        tensor_enable_clang_tidy()
    endif()

    if (TENSOR_CPPCHECK)
        tensor_enable_cppcheck()
    endif()

    if (TENSOR_ADD_SANITIZERS)
        tensor_add_sanitizers()
    endif()

    if (TENSOR_COVERAGE)
        tensor_add_coverage()
    endif()

	# Clang-tidy does not understand G++ precompiled headers
	if (TENSOR_CLANG_TIDY)
	    message(STATUS "CMAKE_CXX_COMPILER_ID=${CMAKE_CXX_COMPILER_ID}")
	    if (CMAKE_CXX_COMPILER_ID MATCHES "(GNU|Clang)")
		    set(TENSOR_USE_PCH OFF)
		endif()
	endif()
	if (TENSOR_USE_PCH)
	    message(STATUS "Using precompiled headers")
        target_precompile_headers(tensor_options
            INTERFACE <algorithm> <cmath> <complex> <cstring> <functional>
                      <memory> <iostream> <string> <vector>)
    endif()
endfunction()

function(tensor_add_coverage)
    if(CMAKE_BUILD_TYPE MATCHES "Debug")
        message(STATUS "Adding coverage options")
        if(CMAKE_CXX_COMPILER_ID MATCHES ".*GNU")
            target_compile_options(tensor_options INTERFACE --coverage)
            target_link_options(tensor_options INTERFACE --coverage)
        else()
            message(STATUS "Unknown code coverage options for ${CMAKE_CXX_COMPILER_ID} C++ Compiler")
        endif()
    endif()
endfunction()

function(tensor_add_sanitizers)
    if(CMAKE_BUILD_TYPE MATCHES "Debug" OR CMAKE_BUILD_TYPE MATCHES "RelWithDebInfo")
        message(STATUS "Adding sanitizer options")
        if(CMAKE_CXX_COMPILER_ID MATCHES ".*(Clang|GNU)")
            target_compile_options(tensor_options INTERFACE $<$<CONFIG:Debug>:-fsanitize=address>)
            target_link_options(tensor_options INTERFACE $<$<CONFIG:Debug>:-fsanitize=address>)
        else()
            string(FIND "$ENV{VSINSTALLDIR}" "$ENV{PATH}" index_of_vs_install_dir)
            if("index_of_vs_install_dir" STREQUAL "-1")
            message(
                SEND_ERROR
                "Using MSVC sanitizers requires setting the MSVC environment before building the project. Please manually open the MSVC command prompt and rebuild the project."
            )
            endif()
            target_compile_options(tensor_options INTERFACE $<$<CONFIG:Debug>:/fsanitize=address /Zi /INCREMENTAL:NO>)
            target_link_options(tensor_options INTERFACE $<$<CONFIG:Debug>:/INFERASANLIBS /INCREMENTAL:NO>)
        endif()
    endif()
endfunction()

function(tensor_add_optimizations)
    if(MSVC)
        set(PROJECT_OPTIMIZATIONS /O2)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
        set(PROJECT_OPTIMIZATIONS -O2)
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set(PROJECT_OPTIMIZATIONS -O2)
    else()
        message(AUTHOR_WARNING "No compiler optimizations set for '${CMAKE_CXX_COMPILER_ID}' compiler.")
    endif()

    message(STATUS "Compiler optimizations ${PROJECT_OPTIMIZATIONS}")
    target_compile_options(tensor_options INTERFACE
        $<$<CONFIG:Release,RelWithDebInfo>:${PROJECT_OPTIMIZATIONS}>)
endfunction()

# Warnings collected from here:
#
# https://github.com/lefticus/cppbestpractices/blob/master/02-Use_the_Tools_Available.md

function(tensor_add_warnings)
    set(MSVC_WARNINGS
        /W4 # Baseline reasonable warnings
        /w14242 # 'identifier': conversion from 'type1' to 'type1', possible loss of data
        /w14254 # 'operator': conversion from 'type1:field_bits' to 'type2:field_bits', possible loss of data
        /w14263 # 'function': member function does not override any base class virtual member function
        /w14265 # 'classname': class has virtual functions, but destructor is not virtual instances of this class may not
                # be destructed correctly
        /w14287 # 'operator': unsigned/negative constant mismatch
        /we4289 # nonstandard extension used: 'variable': loop control variable declared in the for-loop is used outside
                # the for-loop scope
        /w14296 # 'operator': expression is always 'boolean_value'
        /w14311 # 'variable': pointer truncation from 'type1' to 'type2'
        /w14545 # expression before comma evaluates to a function which is missing an argument list
        /w14546 # function call before comma missing argument list
        /w14547 # 'operator': operator before comma has no effect; expected operator with side-effect
        /w14549 # 'operator': operator before comma has no effect; did you intend 'operator'?
        /w14555 # expression has no effect; expected expression with side- effect
        /w14619 # pragma warning: there is no warning number 'number'
        /w14640 # Enable warning on thread un-safe static member initialization
        /w14826 # Conversion from 'type1' to 'type_2' is sign-extended. This may cause unexpected runtime behavior.
        /w14905 # wide string literal cast to 'LPSTR'
        /w14906 # string literal cast to 'LPWSTR'
        /w14928 # illegal copy-initialization; more than one user-defined conversion has been implicitly applied
        /permissive- # standards conformance mode for MSVC compiler.
    )

    set(CLANG_WARNINGS
        -Wall
        -Wextra # reasonable and standard
        -Wshadow # warn the user if a variable declaration shadows one from a parent context
        -Wnon-virtual-dtor # warn the user if a class with virtual functions has a non-virtual destructor. This helps
        # catch hard to track down memory errors
        -Wold-style-cast # warn for c-style casts
        -Wcast-align # warn for potential performance problem casts
        -Wunused # warn on anything being unused
        -Woverloaded-virtual # warn if you overload (not override) a virtual function
        -Wpedantic # warn if non-standard C++ is used
        -Wconversion # warn on type conversions that may lose data
        -Wsign-conversion # warn on sign conversions
        -Wnull-dereference # warn if a null dereference is detected
        -Wdouble-promotion # warn if float is implicit promoted to double
        -Wformat=2 # warn on security issues around functions that format output (ie printf)
        -Wimplicit-fallthrough # warn on statements that fallthrough without an explicit annotation
    )

    set(GCC_WARNINGS
        ${CLANG_WARNINGS}
        -Wmisleading-indentation # warn if indentation implies blocks where blocks do not exist
        -Wduplicated-cond # warn if if / else chain has duplicated conditions
        -Wduplicated-branches # warn if if / else branches have duplicated code
        -Wlogical-op # warn about logical operations being used where bitwise were probably wanted
        -Wuseless-cast # warn if you perform a cast to the same type
    )

    if(WARNINGS_AS_ERRORS)
        message(AUTHOR_WARNING "NOTE: ${WARNINGS_AS_ERRORS}")
        list(APPEND CLANG_WARNINGS -Werror)
        list(APPEND GCC_WARNINGS -Werror)
        list(APPEND MSVC_WARNINGS /WX)
    endif()

    if(MSVC)
        set(PROJECT_WARNINGS ${MSVC_WARNINGS})
    elseif(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
        set(PROJECT_WARNINGS ${CLANG_WARNINGS})
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set(PROJECT_WARNINGS ${GCC_WARNINGS})
    else()
        message(AUTHOR_WARNING "No compiler warnings set for '${CMAKE_CXX_COMPILER_ID}' compiler.")
    endif()

    message(STATUS "Compiler warnings ${PROJECT_WARNINGS}")
    target_compile_options(tensor_options INTERFACE ${PROJECT_WARNINGS})

endfunction()

macro(tensor_enable_clang_tidy)
    find_program(CLANGTIDY clang-tidy)
    if(CLANGTIDY)
        set(CMAKE_CXX_CLANG_TIDY ${CLANGTIDY})
        if(${CMAKE_CXX_STANDARD})
            set(CMAKE_CXX_CLANG_TIDY ${CMAKE_CXX_CLANG_TIDY} -extra-arg=-std=c++${CMAKE_CXX_STANDARD})
        endif()
        if(WARNINGS_AS_ERRORS)
            list(APPEND CMAKE_CXX_CLANG_TIDY -warnings-as-errors=*)
        endif()
		set(CMAKE_CXX_CLANG_TIDY ${CMAKE_CXX_CLANG_TIDY} PARENT_SCOPE)
        message(STATUS "clang-tidy enabled with options CMAKE_CXX_CLANG_TIDY=${CMAKE_CXX_CLANG_TIDY}")
    else()
        message(WARNING "clang-tidy requested but not found")
    endif()
endmacro()

macro(tensor_enable_cppcheck)
    find_program(CPPCHECK cppcheck)
    if(CPPCHECK)
        set(CMAKE_CXX_CPPCHECK ${CPPCHECK}
          --template=${CPPCHECK_TEMPLATE}
          --enable=style,performance,warning,portability
          --inline-suppr
		  --library=googletest
		  -D__cppcheck__
          # We cannot act on a bug/missing feature of cppcheck
          --suppress=internalAstError
          # if a file does not have an internalAstError, we get an unmatchedSuppression error
          --suppress=unmatchedSuppression
          # noExplicitConstructor gives false positives on move constructors
          --suppress=noExplicitConstructor
		  --suppressions-list=${PROJECT_SOURCE_DIR}/.cppcheck
          --inconclusive)
        if(${CMAKE_CXX_STANDARD})
            set(CMAKE_CXX_CPPCHECK ${CMAKE_CXX_CPPCHECK} --std=c++${CMAKE_CXX_STANDARD})
        endif()
        if(WARNINGS_AS_ERRORS)
            list(APPEND CMAKE_CXX_CPPCHECK -warnings-as-errors=*)
        endif()
        set(CMAKE_CXX_CPPCHECK ${CMAKE_CXX_CPPCHECK} PARENT_SCOPE)
        message(STATUS "cppcheck enabled with options CMAKE_CXX_CPPCHECK=${CMAKE_CXX_CPPCHECK}")
    else()
        message(WARNING "cppcheck requested but not found")
    endif()
endmacro()
