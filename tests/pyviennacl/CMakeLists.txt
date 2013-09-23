add_test(NAME blas3_prod_py
  WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/tests/pyviennacl"
  COMMAND python3 blas3_prod.py)

add_test(NAME blas3_solve_py
  WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/tests/pyviennacl"
  COMMAND python3 blas3_solve.py)

add_test(NAME scalar_operations_py
  WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/tests/pyviennacl"
  COMMAND python3 scalar_operations.py)

add_test(NAME vector_operations_py
  WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/tests/pyviennacl"
  COMMAND python3 vector_operations.py)

add_test(NAME matrix_operations_py
  WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/tests/pyviennacl"
  COMMAND python3 matrix_operations.py)

