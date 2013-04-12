include_directories(
  ${PYTHON_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
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
  OUTPUT ${OUTPUT}/stamp
  COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_CURRENT_SOURCE_DIR}/src"
                                             "${CMAKE_CURRENT_BINARY_DIR}/src"
  COMMAND ${PYTHON_EXECUTABLE} setup.py build
  COMMAND ${CMAKE_COMMAND} -E touch ${OUTPUT}/stamp
  DEPENDS ${DEPS}
)

add_custom_target(target ALL DEPENDS ${OUTPUT}/stamp)

install(CODE "execute_process(COMMAND ${PYTHON_EXECUTABLE} ${SETUP_PY} install)")

