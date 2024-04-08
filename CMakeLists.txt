cmake_minimum_required(VERSION 3.27)
project(cat_hipsterizer)

#TF_CFLAGS=$(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
#TF_LFLAGS=$(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')

set(CMAKE_CXX_STANDARD 14)

find_package(Protobuf REQUIRED)
#find_package(TensorFlow REQUIRED)

include_directories(CppVersion)
include_directories(${Protobuf_INCLUDE_DIRS})

#set(PROTO_FILE your_proto_file.proto)
#protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${PROTO_FILE})

add_executable(cat_hipsterizer
        CppVersion/load_model.cpp
        CppVersion/load_model.h
        CppVersion/saved_model.pb.cc
        CppVersion/saved_model.pb.h)
target_link_libraries(cat_hipsterizer ${Protobuf_LIBRARIES})
#target_include_directories(cat_hipsterizer PRIVATE ${TensorFlow_INCLUDE_DIRS})
#target_link_libraries(cat_hipsterizer ${TensorFlow_LIBRARIES})
