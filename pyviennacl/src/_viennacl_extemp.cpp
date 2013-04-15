#include <boost/python.hpp>
#define VIENNACL_WITH_OPENCL
#include <viennacl/vector.hpp>
#include <iostream>

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

#define CHILD_CONST_FUNCTION(R, T, f) \
        R f() const { return static_cast<T const&>(*this).f(); }
#define CHILD_CONST_MEMBER(R, T, m) \
        R get_ ## m () const { return static_cast<T const&>(*this).m; }

namespace pyviennacl
{

  template <class CRTP>
  class VCLVectorBase 
  {
  public:
    typedef std::vector<double>      cpu_vector_type;
    typedef viennacl::vector<double> vcl_vector_type;

    typedef vcl_vector_type::value_type value_type;
    typedef vcl_vector_type::size_type  size_type;

    value_type operator[](size_type i) const
    {
      // Needs optimising
      return static_cast<CRTP const&>(*this)[i];
    }

    CHILD_CONST_MEMBER(vcl_vector_type, CRTP, vector)

    CHILD_CONST_FUNCTION(list, CRTP, get_value)
    CHILD_CONST_FUNCTION(size_type, CRTP, size)
    CHILD_CONST_FUNCTION(vcl_vector_type::const_iterator, CRTP, begin)
    CHILD_CONST_FUNCTION(vcl_vector_type::const_iterator, CRTP, end)

  };

  class VCLVector : public VCLVectorBase<VCLVector>
  {
    typedef typename VCLVectorBase<VCLVector>::cpu_vector_type cpu_vector_type;
    typedef typename VCLVectorBase<VCLVector>::vcl_vector_type vcl_vector_type;

    typedef typename VCLVectorBase<VCLVector>::size_type  size_type;
    typedef typename VCLVectorBase<VCLVector>::value_type value_type;

  public:
    vcl_vector_type vector;

    // Need to reimplement using boost::numpy
    list get_value() const
    {
      list l;
      size_type&& s = vector.size();
      cpu_vector_type temp(s);
      viennacl::fast_copy(vector.begin(), vector.end(),
			  temp.begin());
      for (size_type i = 0; i < s; ++i)
	l.append((double)temp[i]);
      return l;
    }

    size_type size() const
    {
      return vector.size();
    }

    vcl_vector_type::const_iterator begin() const
    {
      return vector.begin();
    }

    vcl_vector_type::const_iterator end() const
    {
      return vector.end();
    }

    VCLVector() {}

    VCLVector(size_type s)
    {
      vector = vcl_vector_type(s);
    }

    template <class CRTP>
    VCLVector(VCLVector const& vcl_v)
    {
      // need an intermediary so as to inhibit infinite regress
      std::cout << vcl_v.size() << std::endl;
      //CRTP const& v = vcl_v;
      //this->vector.resize(v.size());
      //viennacl::copy(v.begin(), v.end(), this->vector.begin());
      vector = vcl_v.vector;
    }

    // Need to reimplement using boost::numpy
    VCLVector(list l)
    {
      size_type s = len(l);
      cpu_vector_type cpu_vector(s);
      for (size_type i=0; i < s; ++i)
	cpu_vector[i] = extract<cpu_vector_type::value_type>(l[i]);
      vector.resize(s);
      viennacl::fast_copy(cpu_vector.begin(), cpu_vector.end(),
			  vector.begin());
    }

    ~VCLVector() {}

    value_type operator[](size_type&& i) const
    {
      return vector[i];
    }

    VCLVector const& operator=(VCLVector const& b)
    {
      vector = b.vector;
      return *this;
    }
      
  };

  template <class VCLVectorL, class VCLVectorR>
  class VCLVectorAdd : public VCLVectorBase<VCLVectorAdd<VCLVectorL, 
							 VCLVectorR> >
  {
    typedef typename VCLVectorBase<VCLVector>::cpu_vector_type cpu_vector_type;
    typedef typename VCLVectorBase<VCLVector>::vcl_vector_type vcl_vector_type;

    typedef typename VCLVectorBase<VCLVector>::size_type  size_type;
    typedef typename VCLVectorBase<VCLVector>::value_type value_type;
    
    VCLVectorBase<VCLVectorL> const& lhs;
    VCLVectorBase<VCLVectorR> const& rhs;
    int accessed = 0;

  public:
    cpu_vector_type cpu_vector;
    vcl_vector_type vector;

    ~VCLVectorAdd() {}

    VCLVectorAdd() 
      : lhs(VCLVectorBase<VCLVectorL>()), 
	rhs(VCLVectorBase<VCLVectorL>())
    {}

    VCLVectorAdd(VCLVectorBase<VCLVectorL> const& _lhs,
		 VCLVectorBase<VCLVectorR> const& _rhs)
      : lhs(_lhs), rhs(_rhs)
    {
      //only do the actual addition when we access the vector
      assert(_lhs.size() == _rhs.size());
    }

    // Need to reimplement using boost::numpy
    list get_value()
    {
      if (!accessed) {
	cpu_vector.resize(lhs.size());
	vector = lhs.get_vector() + rhs.get_vector();
	viennacl::fast_copy(vector.begin(), vector.end(),
			    cpu_vector.begin());
	accessed = 1;
      }

      list l;
      for (size_type i = 0; i < lhs.size(); ++i)
	l.append((double)cpu_vector[i]);
      return l;
    }

    size_type size() const
    {
      return lhs.size();
    }

    value_type operator[](size_type i) const
    {
      if (accessed) {
	return cpu_vector[i];
      } else {
	cpu_vector.resize(lhs.size());
	vector = lhs.get_vector() + rhs.get_vector();
	viennacl::fast_copy(vector.begin(), vector.end(),
			    cpu_vector.begin());
	accessed = 1;
	return cpu_vector[i];
      }      
    }
  };

  template <class L, class R>
  VCLVectorAdd<L, R>
  operator+(VCLVectorBase<L> const& l, 
	    VCLVectorBase<R> const& r)
  {
    return VCLVectorAdd<L, R>(l, r);
  }

  /*
  template <class L, class R>
  VCLVectorAdd<L, R>
  operator+(boost::python::other<L> l, 
	    boost::python::other<R> r)
  {
    return VCLVectorAdd<L, R>(extract<L>(l), extract<R>(r));
  }
  */

  template <class CRTP>
  class VCLVectorBaseConverter {
    PyObject* convert(VCLVectorBase<CRTP> const& v)
    {
      return NULL;
    }
  };

}

BOOST_PYTHON_MODULE(_viennacl_extemp)
{
  using namespace pyviennacl;

  class_<VCLVectorAdd<VCLVectorAdd<VCLVectorAdd<VCLVector, VCLVector>, VCLVector>, VCLVector> >
    ("vector", no_init)
    .def(self + VCLVector())
    .def(VCLVector() + self)
    .def(self + self)
    .add_property("value", &VCLVectorAdd<VCLVectorAdd<VCLVectorAdd<VCLVector, VCLVector>, VCLVector>, VCLVector>::get_value)
    ;

  class_<VCLVectorAdd<VCLVectorAdd<VCLVector, VCLVector>, VCLVector> >
    ("vector", no_init)
    .def(self + VCLVector())
    .def(VCLVector() + self)
    .def(self + self)
    .add_property("value", &VCLVectorAdd<VCLVectorAdd<VCLVector, VCLVector>, VCLVector>::get_value)
    ;

  class_<VCLVectorAdd<VCLVector, VCLVector> >
    ("vector", no_init)
    .def(self + VCLVector())
    .def(VCLVector() + self)
    .def(self + self)
    .add_property("value", &VCLVectorAdd<VCLVector, VCLVector>::get_value)
    ;

  class_<VCLVector>("vector")
    .def(init<int>())
    .def(init<list>())
    //.def(init<boost::python::object>())
    .def(self + self)
    .def(self + VCLVectorAdd<VCLVector, VCLVector>())
    .def(VCLVectorAdd<VCLVector, VCLVector>() + self)
    //.def(vector<viennacl::vector<double>, double>() + self)
    //.def(self += vector<viennacl::vector<double>, double>())
    .add_property("value", &VCLVector::get_value)
    ;

}

