#include <boost/python.hpp>
#define VIENNACL_WITH_OPENCL
#include <viennacl/vector.hpp>

using namespace boost::python;

/*

  TODO: wrap the viennacl vector type and operators,
        in terms of Python's list?..

	But what about the gpu/cpu memory copy problem?...

	Need to make a Pythonic vector class
         - constructed from a list of floats/doubles
	   and, optionally, the memory alignment
	 - supports in-place operations
	 - fast_copy() vs copy() (fast_copy for linear sequence in memory)

        Need to support multiple back-ends
         - build multiple times, for each back-end, with #ifdefs..

 */

/* PUZZLE:

   Compare the execution times of the two vector operations x = y1 + y2;
   and x = y1 + y2 + y1 + y2; for ViennaCL vectors x, y1, and y2 for
   different sizes (from about 100 entries to about 1.000.000
   entries). Plot the curves and explain the differences you observe. You
   are free to use either of the OpenMP, CUDA, or OpenCL backends.

*/

namespace pyviennacl
{

  template <class VectorType, class ScalarType>
  void vector_to_list(VectorType const& v, list *l)
  {
    for(unsigned int i=0; i < v.size(); ++i)
      l->append((ScalarType)v[i]);
  }

  template <class VectorType, class ScalarType>
  void list_to_vector(list const& l, VectorType *v)
  {
    // fill v from l
    // very slow on gpu back-end
    // need a more elegant solution!

    for(unsigned int i=0; i < v->size(); ++i)
      (*v)[i] = extract<ScalarType>(l[i]);
  }

  template <class VectorType, class ScalarType>
  class vector
  {
  
  public:
    int align;
    VectorType *vec = NULL;

    //TODO: constructors/destructor
    //TODO: getter and setters
    //TODO: operator overloads...
    //TODO: Pythonic operators/properties...
    //TODO: memory alignment handling -- meta-templates?
    //TODO: memory domain handling
    //TODO: operator+,+= with list (vs vector)

    list get_value()
    {
      list l;
      vector_to_list<VectorType, ScalarType>((*vec), &l);
      return l;
    }

    void set_value(list l)
    {
      int size = len(l);
      VectorType *v = new VectorType(size);
      list_to_vector<VectorType, ScalarType>(l, v);
      vec = v;
    }

    friend void swap(vector<VectorType, ScalarType> &a,
		     vector<VectorType, ScalarType> &b)
    {
      std::swap(a.align, b.align);
      //viennacl::swap(a.vec, b.vec);
      std::swap(a.vec, b.vec);
    }

    vector<VectorType, ScalarType>&
    operator=(vector<VectorType, ScalarType> b)
    {
      swap(*this, b);
      return *this;
    }
  
    vector<VectorType, ScalarType>&
    operator+=(const vector<VectorType, ScalarType> &y)
    {
      (*vec) += *(y.vec);
      return *this;
    }
  
    vector<VectorType, ScalarType>
    operator+(const vector<VectorType, ScalarType> &y) const
    {
      vector<VectorType, ScalarType> temp(y);
      temp += (*this);
      
      return temp;
    }
    
    vector(const vector<VectorType, ScalarType> &b)
    {
      align = b.align;
      VectorType *v = new VectorType(b.vec->size());

      // assuming that memory is allocated linearly, could fast_copy...
      // currently, assumes very little, but both vectors are OpenCL
      viennacl::copy(*(b.vec), *v);
      vec = v;

    }

    vector(int a=NULL)
    {
      align = a;
    }

    vector(list l, int a=NULL)
    {
      align = a;
      set_value(l);
    }

    ~vector()
    {
      delete vec;
    }
  };
}

BOOST_PYTHON_MODULE(_viennacl)
{

  using namespace pyviennacl;

  class_< vector<viennacl::vector<double>, double> >("vector")
    .def(init<int>())
    .def(init<vector<viennacl::vector<double>, double> >())
    .def(init<list>())
    .def(init<list, int>())
    .def(self + vector<viennacl::vector<double>, double>())
    .def(vector<viennacl::vector<double>, double>() + self)
    .def(self += vector<viennacl::vector<double>, double>())
    .add_property("value",
		  &vector<viennacl::vector<double>, double>::get_value,
		  &vector<viennacl::vector<double>, double>::set_value)
    .def_readonly("align", &vector<viennacl::vector<double>, double>::align);
    ;

}

