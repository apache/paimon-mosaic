# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

cmake_minimum_required(VERSION 3.16)
project(mosaic-cpp-test CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Arrow REQUIRED)

set(MOSAIC_FFI_LIB "" CACHE FILEPATH "Path to libpaimon_mosaic_ffi shared library")
if(NOT MOSAIC_FFI_LIB)
    if(APPLE)
        set(MOSAIC_FFI_LIB "${CMAKE_SOURCE_DIR}/../target/release/libpaimon_mosaic_ffi.dylib")
    elseif(WIN32)
        set(MOSAIC_FFI_LIB "${CMAKE_SOURCE_DIR}/../target/release/paimon_mosaic_ffi.dll")
    else()
        set(MOSAIC_FFI_LIB "${CMAKE_SOURCE_DIR}/../target/release/libpaimon_mosaic_ffi.so")
    endif()
endif()

add_executable(test_mosaic test_mosaic.cpp)
target_include_directories(test_mosaic PRIVATE ${CMAKE_SOURCE_DIR}/../include)
target_link_libraries(test_mosaic Arrow::arrow_shared ${MOSAIC_FFI_LIB})
