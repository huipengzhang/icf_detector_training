cmake_minimum_required(VERSION 2.8)

project(RecognitionForRobot)

find_package(OpenCV 2.4 REQUIRED)
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

file(GLOB src
        "./*.cpp"
        "./*.hpp"
        "./*.pb.cc"
        "./*.pb.h")

file(GLOB src_main_get_multiscale_model
    "./main_get_multiscale_model.cpp")

file(GLOB src_main_train
    "./main_train.cpp")

file(GLOB src_main
    "./main.cpp")

file(GLOB src_main_test
    "./main_test.cpp")

find_package(Threads REQUIRED)
find_package(Protobuf REQUIRED)  # diffrent version problem



list(REMOVE_ITEM src ${src_main_train} ${src_main} ${src_main_get_multiscale_model} ${src_main_test})

add_executable(main ${src} ${src_main})
target_link_libraries(main
        ${OpenCV_LIBS}
        "${CMAKE_THREAD_LIBS_INIT}"
        ${PROTOBUF_LIBRARIES})

add_executable(main_train ${src} ${src_main_train})
target_link_libraries(main_train
        ${OpenCV_LIBS}
        "${CMAKE_THREAD_LIBS_INIT}"
        ${PROTOBUF_LIBRARIES})

add_executable(main_test ${src} ${src_main_test})
target_link_libraries(main_test
        ${OpenCV_LIBS}
        "${CMAKE_THREAD_LIBS_INIT}"
        ${PROTOBUF_LIBRARIES})


add_executable(main_get_multiscale_model ${src} ${src_main_get_multiscale_model})
target_link_libraries(main_get_multiscale_model
        ${OpenCV_LIBS}
        "${CMAKE_THREAD_LIBS_INIT}"
        ${PROTOBUF_LIBRARIES})
