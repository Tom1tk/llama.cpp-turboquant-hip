set(TARGET llama-niah)
add_executable(${TARGET}
    niah.cpp
    pflash.cpp
)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/common
    ${CMAKE_SOURCE_DIR}/vendor
)
target_link_libraries(${TARGET} PRIVATE llama-common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()

if(GGML_HIPBLAS)
    set(TEST_BSA test-bsa)
    add_executable(${TEST_BSA}
        test_bsa.cpp
    )
    target_include_directories(${TEST_BSA} PRIVATE
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/common
        /opt/rocm/include
    )
    target_link_libraries(${TEST_BSA} PRIVATE llama-common llama ggml-hip ${CMAKE_THREAD_LIBS_INIT} amdhip64)
    target_link_directories(${TEST_BSA} PRIVATE /opt/rocm/lib ${CMAKE_BINARY_DIR}/bin)
    target_compile_features(${TEST_BSA} PRIVATE cxx_std_17)
    target_compile_definitions(${TEST_BSA} PRIVATE GGML_USE_HIP __HIP_PLATFORM_AMD__)
endif()
