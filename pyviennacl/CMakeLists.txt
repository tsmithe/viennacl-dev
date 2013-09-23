#SET(BOOST_INCLUDEDIR ${PROJECT_SOURCE_DIR}/external/boost_numpy)
#SET(BOOST_LIBRARYDIR ${PROJECT_BINARY_DIR}/lib)
SET(Boost_USE_STATIC_LIBS OFF)
if (BOOST_COMPONENTS)
  find_package(Boost REQUIRED COMPONENTS ${BOOST_COMPONENTS})
else ()
  find_package(Boost REQUIRED COMPONENTS python-py27)
endif ()

include_directories(
  ${PYTHON_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/external/boost_numpy
  ${NUMPY_INCLUDE_DIRS}
)

get_property(INCLUDE_DIRS DIRECTORY
  ${CMAKE_CURRENT_SOURCE_DIR}
  PROPERTY INCLUDE_DIRECTORIES)

set(SETUP_PY_IN "${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in")
set(SETUP_PY    "${CMAKE_CURRENT_BINARY_DIR}/setup.py")
set(OUTPUT      "${CMAKE_CURRENT_BINARY_DIR}")

configure_file(${SETUP_PY_IN} ${SETUP_PY})

add_custom_command(
  OUTPUT ${OUTPUT}/build-stamp
  COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_CURRENT_SOURCE_DIR}/src"
                                             "${CMAKE_CURRENT_BINARY_DIR}/src"
  COMMAND ${PYTHON_EXECUTABLE} setup.py build
  COMMAND ${CMAKE_COMMAND} -E touch ${OUTPUT}/build-stamp
  DEPENDS ${DEPS}
)

add_custom_command(
  OUTPUT ${OUTPUT}/doc-stamp
  COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_CURRENT_SOURCE_DIR}/doc"
                                             "${CMAKE_CURRENT_BINARY_DIR}/doc"
  COMMAND ${PYTHON_EXECUTABLE} doc/build-doc.py
  COMMAND ${CMAKE_COMMAND} -E touch ${OUTPUT}/doc-stamp
  DEPENDS ${DEPS} build-stamp
)

add_custom_target(target ALL DEPENDS ${OUTPUT}/build-stamp ${OUTPUT}/doc-stamp)

install(CODE "execute_process(COMMAND ${PYTHON_EXECUTABLE} ${SETUP_PY} install)")

