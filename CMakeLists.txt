# Project setup
###############

cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
if(COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)


project(ViennaCL)

set(VERSION_MAJOR 0)
set(VERSION_MINOR 9999)
set(VERSION_PATCH 5)
set(VERSION ${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH})

SET(CMAKE_VERBOSE_MAKEFILE ON)   #uncomment this line to increase CMake verbosity

#SET(BOOSTPATH "C:/Program\ Files\ (x86)/boost/boost_1_42")    #Boost path for other systems - customize as needed

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

# Add visibility of headers
# Necessary for Qt-Creator usage.
################
file( GLOB_RECURSE MAKE_HEADERS_VISIBLE_SRC
    *.hpp *.h)
add_custom_target( MAKE_HEADERS_VISIBLE SOURCES ${MAKE_HEADERS_VISIBLE_SRC} )


# Common options
################

set(VIENNACL_SRC_DIST TRUE)
include(ViennaCLCommon)

# Set release build:
IF(DEFINED CMAKE_BUILD_TYPE)
 SET (CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE})
ELSE()
 SET (CMAKE_BUILD_TYPE Release)
ENDIF()

# User options
##############

option(BUILD_DOXYGEN_DOCS "Build the Doxygen-generated API docs" OFF)

option(BUILD_MANUAL "Build the PDF manual" OFF)

option(BUILD_PYVIENNACL "Build the PyViennaCL Python bindings" ON)

cmake_dependent_option(ENABLE_DIST "Enable the the distribution tarball target" ON "BUILD_DOXYGEN_DOCS;BUILD_MANUAL" ON)

mark_as_advanced(BUILD_DOXYGEN_DOCS BUILD_MANUAL ENABLE_DIST)

# Find prerequisites
####################

if(BUILD_DOXYGEN_DOCS)
   find_package(Doxygen REQUIRED)
endif()

if(BUILD_MANUAL)
   find_package(LATEX REQUIRED)
endif()

if(BUILD_PYVIENNACL)
  if(NOT DEFINED PYTHON_EXECUTABLE)
    set(
      PYTHON_EXECUTABLE "python"
      CACHE STRING
      "Change this to the path for your interpreter; for instance 'python3'")
  endif()
  execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" ${CMAKE_SOURCE_DIR}/pyviennacl/pyversions.py python
    OUTPUT_VARIABLE PYVERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
    )
  execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" ${CMAKE_SOURCE_DIR}/pyviennacl/pyversions.py boost
    OUTPUT_VARIABLE BOOST_COMPONENTS
    OUTPUT_STRIP_TRAILING_WHITESPACE
    )
  #set(Python_ADDITIONAL_VERSIONS ${PYVERSION})
  find_package(PythonInterp ${PYVERSION} EXACT)
  #message(${PYTHON_VERSION_STRING})
  if(DEFINED PYTHON_INCLUDE_DIR AND NOT DEFINED PYTHON_INCLUDE_DIR2)
    set(PYTHON_INCLUDE_DIR2 ${PYTHON_INCLUDE_DIR})
  endif()
  find_package(PythonLibs ${PYTHON_VERSION_STRING} EXACT)
  find_package(Numpy REQUIRED)
endif()

# Compile options
#################

include_directories(BEFORE ${PROJECT_SOURCE_DIR})

IF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  INCLUDE_DIRECTORIES("/opt/local/include")
  set(CMAKE_EXE_LINKER_FLAGS "-framework OpenCL")
ENDIF(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")


# Subdirectories
################

if(BUILD_EXAMPLES)
   add_subdirectory(examples)
endif()

if(BUILD_TESTING)
   INCLUDE(CTest)
   add_subdirectory(tests)
endif()

add_subdirectory(doc)

if(BUILD_PYVIENNACL)
   add_subdirectory(external/boost_numpy)
   add_subdirectory(pyviennacl)
endif()

#add_subdirectory(libviennacl)

# Install
#########

install(DIRECTORY viennacl ${CMAKE_CURRENT_BINARY_DIR}/viennacl
   DESTINATION ${INSTALL_INCLUDE_DIR} COMPONENT dev
   FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")

# Dist target
#############

if(ENABLE_DIST)
   file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/dist")

   set(DISTDIR "${PROJECT_BINARY_DIR}/dist/pyviennacl-${VERSION}-all-arch")
   configure_file(cmake/CMakeLists.txt.in "${DISTDIR}/CMakeLists.txt" @ONLY)

   set(DIST_SOURCES_DIR "${PROJECT_BINARY_DIR}/dist/pyviennacl-${VERSION}")
   configure_file(cmake/CMakeLists.txt.in.src "${DIST_SOURCES_DIR}/CMakeLists.txt" @ONLY)

   #
   # Full package:
   #
   add_custom_target(dist
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${DISTDIR}"

      # PyViennaCL
      COMMAND "${CMAKE_COMMAND}" -E copy_directory
         "${PROJECT_SOURCE_DIR}/pyviennacl" "${DISTDIR}/pyviennacl"

      COMMAND "${CMAKE_COMMAND}" -E copy_if_different
         "${PROJECT_SOURCE_DIR}/LICENSE" "${DISTDIR}/LICENSE"

      COMMAND "${CMAKE_COMMAND}" -E copy_if_different
         "${PROJECT_SOURCE_DIR}/changelog" "${DISTDIR}/changelog"

      COMMAND "${CMAKE_COMMAND}" -E copy_if_different
         "${PROJECT_SOURCE_DIR}/README-PyViennaCL" "${DISTDIR}/README"

      COMMAND "${CMAKE_COMMAND}" -E copy_if_different
         "${PROJECT_SOURCE_DIR}/README-ViennaCL" "${DISTDIR}/README"

      COMMAND "${CMAKE_COMMAND}" -E make_directory "${DISTDIR}/build"
      COMMAND "${CMAKE_COMMAND}" -E copy_if_different
         "${PROJECT_SOURCE_DIR}/build/README" "${DISTDIR}/build/README"

      COMMAND "${CMAKE_COMMAND}" -E copy_directory
         "${PROJECT_SOURCE_DIR}/CL" "${DISTDIR}/CL"

      COMMAND "${CMAKE_COMMAND}" -E make_directory "${DISTDIR}/cmake"

      # NumPy for PyViennaCL
      COMMAND "${CMAKE_COMMAND}" -E copy_if_different
         "${PROJECT_SOURCE_DIR}/cmake/FindNumpy.cmake" "${DISTDIR}/cmake/FindNumpy.cmake"

      COMMAND "${CMAKE_COMMAND}" -E copy_if_different
         "${PROJECT_SOURCE_DIR}/cmake/copymanual.cmake" "${DISTDIR}/cmake/copymanual.cmake"
      COMMAND "${CMAKE_COMMAND}" -E copy_if_different
         "${PROJECT_SOURCE_DIR}/cmake/FindMTL.cmake" "${DISTDIR}/cmake/FindMTL.cmake"
      COMMAND "${CMAKE_COMMAND}" -E copy_if_different
         "${PROJECT_SOURCE_DIR}/cmake/FindOpenCL.cmake" "${DISTDIR}/cmake/FindOpenCL.cmake"
      COMMAND "${CMAKE_COMMAND}" -E copy_if_different
         "${PROJECT_SOURCE_DIR}/cmake/FindViennaProfiler.cmake" "${DISTDIR}/cmake/FindViennaProfiler.cmake"
      COMMAND "${CMAKE_COMMAND}" -E copy_if_different
         "${PROJECT_SOURCE_DIR}/cmake/ViennaCLCommon.cmake" "${DISTDIR}/cmake/ViennaCLCommon.cmake"
      COMMAND "${CMAKE_COMMAND}" -E copy_if_different
         "${PROJECT_SOURCE_DIR}/cmake/ViennaCLConfig.cmake.in" "${DISTDIR}/cmake/ViennaCLConfig.cmake.in"
      COMMAND "${CMAKE_COMMAND}" -E copy_if_different
         "${PROJECT_SOURCE_DIR}/cmake/ViennaCLConfigVersion.cmake.in" "${DISTDIR}/cmake/ViennaCLConfigVersion.cmake.in"

      #COMMAND "${CMAKE_COMMAND}" -E make_directory "${DISTDIR}/doc"
      #COMMAND "${CMAKE_COMMAND}" -E copy_if_different
      #   "${PROJECT_BINARY_DIR}/doc/viennacl.pdf" "${DISTDIR}/doc/viennacl.pdf"
      #COMMAND "${CMAKE_COMMAND}" -E copy_directory
      #   "${CMAKE_CURRENT_BINARY_DIR}/doc/doxygen"
      #   "${DISTDIR}/doc/doxygen"

      #COMMAND "${CMAKE_COMMAND}" -E copy_if_different
      #   "${PROJECT_SOURCE_DIR}/doc/Doxyfile.in" "${DISTDIR}/doc/Doxyfile.in"

      #COMMAND "${CMAKE_COMMAND}" -E copy_directory
      #   "${PROJECT_SOURCE_DIR}/examples" "${DISTDIR}/examples"

      #COMMAND "${CMAKE_COMMAND}" -E copy_directory
      #   "${PROJECT_SOURCE_DIR}/libviennacl" "${DISTDIR}/libviennacl"

      COMMAND "${CMAKE_COMMAND}" -E copy_directory
         "${PROJECT_SOURCE_DIR}/external" "${DISTDIR}/external"

      COMMAND "${CMAKE_COMMAND}" -E make_directory "${DISTDIR}/tests"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${DISTDIR}/tests/src"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${DISTDIR}/tests/pyviennacl"
      COMMAND "${CMAKE_COMMAND}" -E copy_directory
         "${PROJECT_SOURCE_DIR}/tests/src" "${DISTDIR}/tests/src"
      COMMAND "${CMAKE_COMMAND}" -E copy_directory
         "${PROJECT_SOURCE_DIR}/tests/pyviennacl" "${DISTDIR}/tests/pyviennacl"
      COMMAND "${CMAKE_COMMAND}" -E copy_if_different
         "${PROJECT_SOURCE_DIR}/tests/CMakeLists.txt" "${DISTDIR}/tests/CMakeLists.txt"

      COMMAND "${CMAKE_COMMAND}" -E copy_directory
         "${PROJECT_SOURCE_DIR}/viennacl" "${DISTDIR}/viennacl"
      COMMAND "${CMAKE_COMMAND}" -E copy_directory
         "${CMAKE_CURRENT_BINARY_DIR}/viennacl/linalg/kernels"
         "${DISTDIR}/viennacl/linalg/kernels"

      COMMAND "${CMAKE_COMMAND}" -E tar czf pyviennacl-${VERSION}-all-arch.tar.gz
           pyviennacl-${VERSION}-all-arch
      WORKING_DIRECTORY "${PROJECT_BINARY_DIR}/dist"
      COMMENT "Generating dist/pyviennacl-${VERSION}-all-arch.tar.gz"
      VERBATIM)
   add_dependencies(dist kernels apidoc manual)

   #
   # Sources only (no generated files):
   #
   foreach(f LICENSE
             README-ViennaCL
             README-PyViennaCL
             changelog
             cmake/copymanual.cmake
             cmake/FindNumpy.cmake # NumPy for PyViennaCL
             cmake/FindMTL.cmake
             cmake/FindOpenCL.cmake
             cmake/FindViennaProfiler.cmake
             cmake/ViennaCLCommon.cmake
             cmake/ViennaCLConfig.cmake.in
             cmake/ViennaCLConfigVersion.cmake.in
             build/README)
      configure_file(${f} "${DIST_SOURCES_DIR}/${f}" COPYONLY)
   endforeach()

   add_custom_target(copymanual-src ALL
      "${CMAKE_COMMAND}"
         -DSRC="${PROJECT_SOURCE_DIR}/doc/manual/"
         -DDST="${DIST_SOURCES_DIR}/doc/manual/"
         -P ${CMAKE_SOURCE_DIR}/cmake/copymanual.cmake
   )

   add_custom_target(dist-src

      # Set up directory layout:
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${DIST_SOURCES_DIR}"

      # PyViennaCL
      COMMAND "${CMAKE_COMMAND}" -E copy_directory
         "${PROJECT_SOURCE_DIR}/pyviennacl" "${DIST_SOURCES_DIR}/pyviennacl"

      COMMAND "${CMAKE_COMMAND}" -E remove_directory "${DIST_SOURCES_DIR}/pyviennacl/debian"
      #COMMAND "${CMAKE_COMMAND}" -E copy_directory
      #   "${PROJECT_SOURCE_DIR}/pyviennacl/debian" "${DIST_SOURCES_DIR}/debian"

      COMMAND "${CMAKE_COMMAND}" -E copy_directory
         "${PROJECT_SOURCE_DIR}/CL" "${DIST_SOURCES_DIR}/CL"

      #COMMAND "${CMAKE_COMMAND}" -E copy_directory
      #   "${PROJECT_SOURCE_DIR}/examples" "${DIST_SOURCES_DIR}/examples"

      #COMMAND "${CMAKE_COMMAND}" -E copy_directory
      #   "${PROJECT_SOURCE_DIR}/libviennacl" "${DIST_SOURCES_DIR}/libviennacl"

      COMMAND "${CMAKE_COMMAND}" -E make_directory "${DIST_SOURCES_DIR}/tests"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${DIST_SOURCES_DIR}/tests/src"
      COMMAND "${CMAKE_COMMAND}" -E copy_directory
         "${PROJECT_SOURCE_DIR}/tests/src" "${DIST_SOURCES_DIR}/tests/src"

      # PyViennaCL
      COMMAND "${CMAKE_COMMAND}" -E copy_directory
         "${PROJECT_SOURCE_DIR}/tests/pyviennacl"
         "${DIST_SOURCES_DIR}/tests/pyviennacl"

      COMMAND "${CMAKE_COMMAND}" -E copy_if_different
         "${PROJECT_SOURCE_DIR}/tests/CMakeLists.txt" "${DIST_SOURCES_DIR}/tests/CMakeLists.txt"

      COMMAND "${CMAKE_COMMAND}" -E copy_directory
         "${PROJECT_SOURCE_DIR}/external" "${DIST_SOURCES_DIR}/external"

      #COMMAND "${CMAKE_COMMAND}" -E copy_directory
      #   "${PROJECT_SOURCE_DIR}/doc/manual/figures" "${DIST_SOURCES_DIR}/doc/manual/figures"

      #COMMAND "${CMAKE_COMMAND}" -E copy_if_different
      #   "${PROJECT_SOURCE_DIR}/doc/Doxyfile.in" "${DIST_SOURCES_DIR}/doc/Doxyfile.in"

      #COMMAND "${CMAKE_COMMAND}" -E copy_if_different
      #   "${PROJECT_SOURCE_DIR}/doc/CMakeLists.txt" "${DIST_SOURCES_DIR}/doc/CMakeLists.txt"

      COMMAND "${CMAKE_COMMAND}" -E copy_directory
         "${PROJECT_SOURCE_DIR}/viennacl" "${DIST_SOURCES_DIR}/viennacl"

      # Create archive:
      COMMAND tar cJf pyviennacl_${VERSION}.orig.tar.xz
           pyviennacl-${VERSION}
      WORKING_DIRECTORY "${PROJECT_BINARY_DIR}/dist"
      COMMENT "Generating dist/pyviennacl_${VERSION}.orig.tar.xz"
      VERBATIM)
   add_dependencies(dist-src copymanual-src)

endif()
