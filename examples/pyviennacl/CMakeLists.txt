include_directories(
  ${PYTHON_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/external/boost_numpy
  ${NUMPY_INCLUDE_DIRS}
)

foreach(tut bugfix)
   add_executable(${tut} ${tut}.cpp)
   if (ENABLE_OPENCL)
     target_link_libraries(${tut} ${OPENCL_LIBRARIES} boost_python-py33 boost_numpy)
     set_target_properties(${tut} PROPERTIES COMPILE_FLAGS "-DVIENNACL_WITH_OPENCL -ggdb")
   endif (ENABLE_OPENCL)
endforeach()
