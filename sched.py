import _viennacl as v # Import low-level viennacl wrapper

a = v.vector(10)     # Set up empty vector to store result
b = v.vector(10,0.1) 
c = v.vector(10,0.2)
d = v.vector(10,0.3)
e = v.vector(10,0.4)

# Set up {0} := [a = {1}] statement node
node0 = v.statement_node(
    v.operation_node_type_family.OPERATION_BINARY_TYPE_FAMILY,
    v.operation_node_type.OPERATION_BINARY_ASSIGN_TYPE,
    v.statement_node_type_family.VECTOR_TYPE_FAMILY,
    v.statement_node_type.VECTOR_DOUBLE_TYPE,
    v.statement_node_type_family.COMPOSITE_OPERATION_FAMILY,
    v.statement_node_type.COMPOSITE_OPERATION_TYPE)
node0.set_lhs_vector_double(a)
node0.set_rhs_node_index(1)

# Set up {1} := [{2} + {3}] statement node
node1 = v.statement_node(
    v.operation_node_type_family.OPERATION_BINARY_TYPE_FAMILY,
    v.operation_node_type.OPERATION_BINARY_ADD_TYPE,
    v.statement_node_type_family.COMPOSITE_OPERATION_FAMILY,
    v.statement_node_type.COMPOSITE_OPERATION_TYPE,
    v.statement_node_type_family.COMPOSITE_OPERATION_FAMILY,
    v.statement_node_type.COMPOSITE_OPERATION_TYPE)
node1.set_lhs_node_index(2)
node1.set_rhs_node_index(3)

# Set up {2} := [b + c] statement node
node2 = v.statement_node(
    v.operation_node_type_family.OPERATION_BINARY_TYPE_FAMILY,
    v.operation_node_type.OPERATION_BINARY_ADD_TYPE,
    v.statement_node_type_family.VECTOR_TYPE_FAMILY,
    v.statement_node_type.VECTOR_DOUBLE_TYPE,
    v.statement_node_type_family.VECTOR_TYPE_FAMILY,
    v.statement_node_type.VECTOR_DOUBLE_TYPE)
node2.set_lhs_vector_double(b)
node2.set_rhs_vector_double(c)

# Set up {3} := [d + e] statement node
node3 = v.statement_node(
    v.operation_node_type_family.OPERATION_BINARY_TYPE_FAMILY,
    v.operation_node_type.OPERATION_BINARY_ADD_TYPE,
    v.statement_node_type_family.VECTOR_TYPE_FAMILY,
    v.statement_node_type.VECTOR_DOUBLE_TYPE,
    v.statement_node_type_family.VECTOR_TYPE_FAMILY,
    v.statement_node_type.VECTOR_DOUBLE_TYPE)
node3.set_lhs_vector_double(d)
node3.set_rhs_vector_double(e)

s = v.statement() # Empty statement object
s.insert_at_begin(node3) # Add nodes to statement tree in correct order
s.insert_at_begin(node2)
s.insert_at_begin(node1)
s.insert_at_begin(node0)

# Make sure we got all the nodes
print("Number of nodes in statement: ", s.size)

# Print out node properties -- just to check we got them right
def print_node(node):
    print(node, "\n",
          node.vcl_statement_node.op_family, "\n",
          node.vcl_statement_node.op_type, "\n",
          node.vcl_statement_node.lhs_type_family, "\n",
          node.vcl_statement_node.lhs_type, "\n",
          node.vcl_statement_node.rhs_type_family, "\n",
          node.vcl_statement_node.rhs_type)
print_node(s.get_node(0))
print_node(s.get_node(1))
print_node(s.get_node(2))
print_node(s.get_node(3))

# Execute the statement
s.execute()

# Display the result
print("Result: ", a.as_ndarray())
