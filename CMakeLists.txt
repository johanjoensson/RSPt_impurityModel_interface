cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(impurityModel_interface VERSION 1.0.0 LANGUAGES C CXX)

include(CMakePrintHelpers)

option(NODEPS "Do not install requirements (impurityModel)" OFF)
option(EDITABLE_DEPS "Install editable dependencies (impurityModel)" OFF)
option(PARALLEL_IMPURITYMODEL "Build the threaded version of impurityModel (IMPURITYMODEL_PARALLEL=1)" OFF)

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

add_custom_command(OUTPUT run_impurityModel.c COMMAND cp ${PROJECT_SOURCE_DIR}/src/run_impurityModel.h . COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/library_builder.py )

if (${NODEPS})
        set(PIP_NODEPS "--no-deps")
else()
        set(PIP_NODEPS)
endif()

if (${EDITABLE_DEPS})
        set(PIP_EDITABLE_DEPS "--editable --src ${PROJECT_SOURCE_DIR}/dependencies/")
else()
        set(PIP_EDITABLE_DEPS )
endif()

if (${PARALLEL_IMPURITYMODEL})
        set(IMPURITYMODEL_ENV "IMPURITYMODEL_PARALLEL=1")
else()
        set(IMPURITYMODEL_ENV )
endif()


# Pass CMAKE compilers and compiler flags on to the installation of dependencies (most importantly impurityModel)
add_custom_target(PYTHON_MODULE COMMAND ${IMPURITYMODEL_ENV} CC=${CMAKE_C_COMPILER} CFLAGS=${CMAKE_C_FLAGS} CXX=${CMAKE_CXX_COMPILER} CXXFLAGS=${CMAKE_CXX_FLAGS} ${Python3_EXECUTABLE} -m pip install ${PIP_NODEPS} ${PIP_USER} ${PROJECT_SOURCE_DIR} )

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
