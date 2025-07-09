cmake_minimum_required(VERSION 3.30)
project(Pong_AI)

set(CMAKE_CXX_STANDARD 20)

include_directories(${CMAKE_SOURCE_DIR}/external/SDL3/include)
link_directories(${CMAKE_SOURCE_DIR}/external/SDL3/lib)

include_directories(${CMAKE_SOURCE_DIR}/external/SDL_ttf/include)
link_directories(${CMAKE_SOURCE_DIR}/external/SDL_ttf/lib)

add_executable(Play
        src/main.cpp
        src/Game.cpp src/Game.h
        src/Paddle.cpp src/Paddle.h
        src/Ball.cpp src/Ball.h
        src/model_serializer.h
        src/model_serializer.cpp
        src/neural_network.h
        src/nn_activation.h
        src/nn_dense.h
        src/nn_interfaces.h
        src/nn_loss.h
        src/nn_optimizer.h
        src/tensor.h
)

target_link_libraries(Play
        ${CMAKE_SOURCE_DIR}/external/SDL3/lib/libSDL3.dll.a
        ${CMAKE_SOURCE_DIR}/external/SDL_ttf/lib/libSDL3_ttf.dll.a
)


add_custom_command(TARGET Play POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_SOURCE_DIR}/external/SDL3/bin/SDL3.dll
        $<TARGET_FILE_DIR:Play>)

set_target_properties(Play PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
)

add_executable(Trainer src/trainer.cpp
        src/neural_network.h
        src/nn_activation.h
        src/nn_dense.h
        src/nn_interfaces.h
        src/nn_loss.h
        src/nn_optimizer.h
        src/tensor.h
        src/model_serializer.h
        src/model_serializer.cpp
        src/Game.h src/Game.cpp
        src/Paddle.h src/Paddle.cpp
        src/Ball.h src/Ball.cpp
        src/training_observer.h
)

target_link_libraries(Trainer
        ${CMAKE_SOURCE_DIR}/external/SDL3/lib/libSDL3.dll.a
        ${CMAKE_SOURCE_DIR}/external/SDL_ttf/lib/libSDL3_ttf.dll.a
)

add_custom_command(TARGET Trainer POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_SOURCE_DIR}/external/SDL3/bin/SDL3.dll
        $<TARGET_FILE_DIR:Trainer>)
add_custom_command(TARGET Trainer POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory
        $<TARGET_FILE_DIR:Trainer>/data
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/data
        $<TARGET_FILE_DIR:Trainer>/data
)

set_target_properties(Trainer PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
)

add_custom_command(TARGET Trainer POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_SOURCE_DIR}/external/SDL3/bin/SDL3.dll"
        "${CMAKE_SOURCE_DIR}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_SOURCE_DIR}/external/SDL_ttf/bin/SDL3_ttf.dll"
        "${CMAKE_SOURCE_DIR}"
)