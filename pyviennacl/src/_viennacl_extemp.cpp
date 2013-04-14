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

/*

  TODO: optimisation -- templates specialisation inheritance,
                        rvalue references and move constructors,
			expression templates
			other c++11 stuff?
  TODO: Pythonic operators/properties...
  TODO: memory alignment handling -- meta-templates?
  TODO: memory domain handling
  TODO: operator+,+= with list (vs vector)

 */

namespace pyviennacl
{

  template <class ScalarType>
  void copy_vector(viennacl::vector<ScalarType> const* a,
		   viennacl::vector<ScalarType> *b)
  {
    viennacl::copy(a->begin(), a->end(), b->begin());
  }

  template <class ScalarType>
  void copy_vector(viennacl::vector<ScalarType> const* a,
		   std::vector<ScalarType> *b)
  {
    viennacl::fast_copy(a->begin(), a->end(), b->begin());
  }


  template <class ScalarType>
  void copy_vector(std::vector<ScalarType> const* a,
		   viennacl::vector<ScalarType> *b)
  {
    viennacl::fast_copy(a->begin(), a->end(), b->begin());
  }


  template <class ScalarType>
  void copy_vector(std::vector<ScalarType> const* a,
		   std::vector<ScalarType> *b)
  {
    std::copy(a->begin(), a->end(), b->begin());
  }

  template <class ScalarType>
  void vector_to_list(viennacl::vector<ScalarType> const& v, list *l)
  {
    std::vector<ScalarType> temp(v.size());
    //viennacl::fast_copy(v.begin(), v.end(), temp.begin());
    copy_vector<ScalarType>(&v, &temp);

    for(unsigned int i=0; i < temp.size(); ++i)
      l->append((ScalarType)temp[i]);
  }

  template <class ScalarType>
  void vector_to_list(std::vector<ScalarType> const& v, list *l)
  {
    for(unsigned int i=0; i < v.size(); ++i)
      l->append((ScalarType)v[i]);
  }

  template <class ScalarType>
  void list_to_vector(list const& l, std::vector<ScalarType> *v)
  {
    for(unsigned int i=0; i < v->size(); ++i)
      (*v)[i] = extract<ScalarType>(l[i]);
  }

  template <class ScalarType>
  void list_to_vector(list const& l, viennacl::vector<ScalarType> *v)
  {
    std::vector<ScalarType> temp(v->size());

    for(unsigned int i=0; i < v->size(); ++i)
      temp[i] = extract<ScalarType>(l[i]);

    //viennacl::fast_copy(temp.begin(), temp.end(), v->begin());
    copy_vector<ScalarType>(&temp, v);
  }

  template <class VectorType, class ScalarType>
  class vector
  {
 
  public:
    int align;
    VectorType *vec = NULL;

    list get_value()
    {
      list l;
      vector_to_list<ScalarType>((*vec), &l);
      return l;
    }

    void set_value(list l)
    {
      int size = len(l);
      VectorType *v = new VectorType(size);
      list_to_vector<ScalarType>(l, v);
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
      //viennacl::copy(b.vec->begin(), b.vec->end(), v->begin());
      copy_vector<ScalarType>(b.vec, v);
      vec = v;
    }

    vector(int a=NULL)
    {
      align = 0; // a;
    }

    vector(list l, int a=NULL)
    {
      align = 0; // a;
      set_value(l);
    }

    ~vector()
    {
      delete vec;
    }
  };
}

BOOST_PYTHON_MODULE(_viennacl_extemp)
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
    .def_readonly("align", &vector<viennacl::vector<double>, double>::align)
    ;

  class_< vector<std::vector<double>, double> >("std_vector")
    .def(init<int>())
    .def(init<vector<std::vector<double>, double> >())
    .def(init<list>())
    .def(init<list, int>())
    //.def(self + vector<std::vector<double>, double>())
    //.def(vector<std::vector<double>, double>() + self)
    //.def(self += vector<std::vector<double>, double>())
    .add_property("value",
		  &vector<std::vector<double>, double>::get_value,
		  &vector<std::vector<double>, double>::set_value)
    .def_readonly("align", &vector<std::vector<double>, double>::align)
    ;

}

