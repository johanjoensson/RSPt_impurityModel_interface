cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(impurityModel_interface VERSION 1.0.0 LANGUAGES C)

include(CMakePrintHelpers)

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
                RESULT_VARIABLE CFFI_INSTALL
                COMMAND_ERROR_IS_FATAL ANY
        )
endif()

execute_process(
        COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/library_builder.py
        RESULT_VARIABLE LIBRARY_BUILT
        COMMAND_ERROR_IS_FATAL ANY
)

execute_process(
        COMMAND ${Python3_EXECUTABLE} -m pip install ${PIP_USER} ${PROJECT_SOURCE_DIR}
        COMMAND_ERROR_IS_FATAL ANY
)

Python3_add_library(${PROJECT_NAME} SHARED ${PROJECT_SOURCE_DIR}/src/run_impurityModel.c)
target_include_directories(${PROJECT_NAME} PRIVATE  ${PROJECT_SOURCE_DIR}/src)
