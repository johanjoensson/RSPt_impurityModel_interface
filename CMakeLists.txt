cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(impurityModel_interface VERSION 1.0.0 LANGUAGES C CXX)

include(CMakePrintHelpers)

option(NODEPS "Do not install requirements (impurityModel)" OFF)
option(EDITABLE_DEPS "Install editable dependencies (impurityModel)" OFF)
option(PARALLEL_IMPURITYMODEL "Build the threaded version of impurityModel (IMPURITYMODEL_PARALLEL=1)" OFF)

# Default to an optimized build of the interface library itself.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
        set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# impurityModel's setup.py reads IMPURITYMODEL_BUILD (release | debug | safe). Set it
# explicitly so the mode does not depend on whatever the calling shell happens to export.
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(_IMPURITYMODEL_BUILD_DEFAULT debug)
else()
        set(_IMPURITYMODEL_BUILD_DEFAULT release)
endif()
set(IMPURITYMODEL_BUILD ${_IMPURITYMODEL_BUILD_DEFAULT} CACHE STRING "impurityModel build mode: release, debug or safe")
set_property(CACHE IMPURITYMODEL_BUILD PROPERTY STRINGS release debug safe)
if(NOT IMPURITYMODEL_BUILD MATCHES "^(release|debug|safe)$")
        message(FATAL_ERROR "IMPURITYMODEL_BUILD must be release, debug or safe (got '${IMPURITYMODEL_BUILD}')")
endif()
message(STATUS "impurityModel build mode: ${IMPURITYMODEL_BUILD}")

set(Python3_FIND_VIRTUALENV FIRST)
find_package(Python3 3.2 COMPONENTS Interpreter Development.Embed  REQUIRED)

if(DEFINED ENV{VIRTUAL_ENV} OR DEFINED ENV{CONDA_PREFIX})
        set(PIP_USER)
else()
        set(PIP_USER "--user")
endif()

execute_process(
        COMMAND ${Python3_EXECUTABLE} -m pip show cffi
        RESULT_VARIABLE CFFI_FOUND
)


if ( CFFI_FOUND EQUAL 0 )
        message(STATUS "Found cffi")
else()
        message(STATUS "Installing cffi")
        execute_process(
                COMMAND ${Python3_EXECUTABLE} -m pip install ${PIP_USER} cffi
                COMMAND_ERROR_IS_FATAL ANY
        )
endif()

# DEPENDS is load bearing: without it this rule has no prerequisites, so editing the header
# or the builder does not regenerate run_impurityModel.c on an incremental build and the
# change silently does not reach the shared library.
add_custom_command(OUTPUT run_impurityModel.c
        COMMAND cp ${PROJECT_SOURCE_DIR}/src/run_impurityModel.h .
        COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/library_builder.py
        DEPENDS ${PROJECT_SOURCE_DIR}/src/run_impurityModel.h
                ${PROJECT_SOURCE_DIR}/library_builder.py
)

if (${NODEPS})
        set(PIP_NODEPS "--no-deps")
else()
        set(PIP_NODEPS)
endif()

# EDITABLE_DEPS: install local checkouts of impurityModel and rspt2spectra in editable mode
# instead of the GitHub copies pinned in pyproject.toml. Point the *_SOURCE_DIR variables at
# existing working trees; a directory that does not exist is cloned from the fork at configure
# time.
set(IMPURITYMODEL_SOURCE_DIR "${PROJECT_SOURCE_DIR}/dependencies/impurityModel" CACHE PATH
        "Local impurityModel checkout used when EDITABLE_DEPS=ON")
set(RSPT2SPECTRA_SOURCE_DIR "${PROJECT_SOURCE_DIR}/dependencies/rspt2spectra" CACHE PATH
        "Local rspt2spectra checkout used when EDITABLE_DEPS=ON")
set(IMPURITYMODEL_GIT_URL "https://github.com/johanjoensson/impurityModel.git" CACHE STRING
        "Repository cloned into IMPURITYMODEL_SOURCE_DIR if it does not exist")
set(RSPT2SPECTRA_GIT_URL "https://github.com/johanjoensson/rspt2spectra.git" CACHE STRING
        "Repository cloned into RSPT2SPECTRA_SOURCE_DIR if it does not exist")

set(PIP_EDITABLE_DEPS)
if (${EDITABLE_DEPS})
        find_package(Git REQUIRED)
        foreach(dep IMPURITYMODEL RSPT2SPECTRA)
                set(dep_dir "${${dep}_SOURCE_DIR}")
                if(NOT EXISTS "${dep_dir}")
                        message(STATUS "Cloning ${${dep}_GIT_URL} into ${dep_dir}")
                        execute_process(
                                COMMAND ${GIT_EXECUTABLE} clone ${${dep}_GIT_URL} ${dep_dir}
                                COMMAND_ERROR_IS_FATAL ANY
                        )
                elseif(NOT EXISTS "${dep_dir}/pyproject.toml")
                        message(FATAL_ERROR "${dep}_SOURCE_DIR=${dep_dir} is not a Python project (no pyproject.toml)")
                endif()
                message(STATUS "Editable ${dep}: ${dep_dir}")
                list(APPEND PIP_EDITABLE_DEPS --editable ${dep_dir})
        endforeach()
endif()

if (${PARALLEL_IMPURITYMODEL})
        set(IMPURITYMODEL_ENV "IMPURITYMODEL_PARALLEL=1")
else()
        set(IMPURITYMODEL_ENV )
endif()


# Forward CFLAGS/CXXFLAGS only when set: setuptools *replaces* Python's own flags with
# them, so an empty value would silently drop e.g. -DNDEBUG from the extension builds.
set(COMPILER_FLAGS_ENV)
if(CMAKE_C_FLAGS)
        list(APPEND COMPILER_FLAGS_ENV "CFLAGS=${CMAKE_C_FLAGS}")
endif()
if(CMAKE_CXX_FLAGS)
        list(APPEND COMPILER_FLAGS_ENV "CXXFLAGS=${CMAKE_CXX_FLAGS}")
endif()

# Pass CMAKE compilers and compiler flags on to the installation of dependencies (most importantly impurityModel)
set(PIP_INSTALL ${IMPURITYMODEL_ENV} IMPURITYMODEL_BUILD=${IMPURITYMODEL_BUILD} CC=${CMAKE_C_COMPILER} ${COMPILER_FLAGS_ENV} CXX=${CMAKE_CXX_COMPILER} ${Python3_EXECUTABLE} -m pip install ${PIP_USER})

if (${EDITABLE_DEPS})
        # The editable installs pull in their own requirements; the interface itself must then be
        # installed with --no-deps, otherwise pip resolves the git URLs in pyproject.toml and
        # replaces the editable installs with the GitHub copies. cffi, the one requirement of the
        # interface that neither dependency provides, is ensured at configure time above.
        add_custom_target(PYTHON_MODULE
                COMMAND ${PIP_INSTALL} ${PIP_NODEPS} ${PIP_EDITABLE_DEPS}
                COMMAND ${PIP_INSTALL} --no-deps ${PROJECT_SOURCE_DIR}
        )
else()
        add_custom_target(PYTHON_MODULE COMMAND ${PIP_INSTALL} ${PIP_NODEPS} ${PROJECT_SOURCE_DIR})
endif()

Python3_add_library(${PROJECT_NAME} SHARED run_impurityModel.c)

# Ensure that the directory containing the Python library is added to the RPATH
# so that consumers of the shared library don't need to manually configure library paths.
set_target_properties(${PROJECT_NAME} PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH_USE_LINK_PATH TRUE
)

# Explicitly append the Python library directories to RPATH to ensure they are
# included even if they are located in system-default directories.
if(Python3_RUNTIME_LIBRARY_DIRS)
    set_property(TARGET ${PROJECT_NAME} APPEND PROPERTY INSTALL_RPATH "${Python3_RUNTIME_LIBRARY_DIRS}")
elseif(Python3_LIBRARY_DIRS)
    set_property(TARGET ${PROJECT_NAME} APPEND PROPERTY INSTALL_RPATH "${Python3_LIBRARY_DIRS}")
endif()
# target_include_directories(${PROJECT_NAME} PRIVATE  ${PROJECT_SOURCE_DIR}/src)
add_dependencies(${PROJECT_NAME} PYTHON_MODULE)

install(TARGETS ${PROJECT_NAME})
