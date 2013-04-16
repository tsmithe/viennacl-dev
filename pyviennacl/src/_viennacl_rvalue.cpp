#include <boost/python.hpp>
#define VIENNACL_WITH_OPENCL
#include <viennacl/vector.hpp>

using namespace boost::python;

/*

  TODO: vector.ones initialiser..
  TODO: optimisation -- templates specialisation inheritance,
                        rvalue references and move constructors,
			expression templates
			other c++11 stuff?
  TODO: Pythonic operators/properties...
  TODO: memory alignment handling -- meta-templates?
  TODO: memory domain handling
  TODO: operator*,*=
  TODO: operator+,+= with list (vs vector)?

 */

namespace pyviennacl
{

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
      *vec += *(y.vec);
      return *this;
    }

    
    vector<VectorType, ScalarType>
    operator+(const vector<VectorType, ScalarType> &y) const
    {
      vector<VectorType, ScalarType> temp(y);
      temp += (*this);
      return temp;
    }
    

    vector(vector<VectorType, ScalarType> &&b)
      : vec(b.vec)
    {
      b.vec = NULL;
    }
    
    vector(const vector<VectorType, ScalarType> &b)
    {
      VectorType *v = new VectorType(b.vec->size());
      copy_vector<ScalarType>(b.vec, v);
      vec = v;
    }
    
    vector()
    {
    }

    vector(list l)
    {
      set_value(l);
    }

    ~vector()
    {
      delete vec;
    }
  };
}

BOOST_PYTHON_MODULE(_viennacl_rvalue)
{

  using namespace pyviennacl;

  class_< vector<viennacl::vector<double>, double> >("vector")
    .def(init<vector<viennacl::vector<double>, double> >())
    .def(init<list>())
    .def(self + self)
    .def(self += self)
    //.def(self + vector<std::vector<double>, double>())
    //.def(self += vector<std::vector<double>, double>())
    .add_property("value",
		  &vector<viennacl::vector<double>, double>::get_value,
		  &vector<viennacl::vector<double>, double>::set_value)
    ;

  /*
  class_< vector<std::vector<double>, double> >("std_vector")
    //.def(init<vector<std::vector<double>, double> >())
    .def(init<list>())
    //.def(self + vector<std::vector<double>, double>())
    //.def(vector<std::vector<double>, double>() + self)
    //.def(self += vector<std::vector<double>, double>())
    .add_property("value",
		  &vector<std::vector<double>, double>::get_value,
		  &vector<std::vector<double>, double>::set_value)
    ;
  */
}

