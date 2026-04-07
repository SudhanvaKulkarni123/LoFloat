# find all .cpp files in test/
file(GLOB TEST_SOURCES "*.cpp")

foreach(test_src ${TEST_SOURCES})
    # turn "test_foo.cpp" into "test_foo"
    get_filename_component(test_name ${test_src} NAME_WE)
    
    add_executable(${test_name} ${test_src})
    target_link_libraries(${test_name} PRIVATE lo_float)
    
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()