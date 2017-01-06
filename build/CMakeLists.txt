cmake_minimum_required(VERSION 2.6)
project(libNN7)

include_directories("../src" "../lib/" "../lib/pugixml")

set(CMAKE_CXX_FLAGS "-std=c++0x")

file(GLOB SOURCE_LIB "../src/*.cpp" "../lib/pugixml/*.cpp")

# cmake -G "MinGW Makefiles" -DBUILD_TESTS=1
IF(BUILD_TESTS)
	set(TEST_SOURCES "../tests")
	add_executable(LinearNeuron_test ${TEST_SOURCES}/NN7LinearNeuron_test.cpp ${SOURCE_LIB})
  add_executable(FeedForward_test ${TEST_SOURCES}/NN7FeedForward_test.cpp ${SOURCE_LIB})
	add_executable(DataVector_test ${TEST_SOURCES}/NN7DataVector_test.cpp ${SOURCE_LIB})
	add_executable(Randomizer_test ${TEST_SOURCES}/NN7Randomizer_test.cpp ${SOURCE_LIB})
  add_executable(SigmoidNeuron_test ${TEST_SOURCES}/NN7SigmoidNeuron_test.cpp ${SOURCE_LIB})
	add_executable(XMLConfig_test ${TEST_SOURCES}/XMLConfig_test.cpp ${SOURCE_LIB})
ENDIF()
