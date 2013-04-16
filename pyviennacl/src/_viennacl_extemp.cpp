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
#define CHILD_FUNCTION(R, T, f) \
  R& f() { return std::forward<T&>(*this).f(); }
#define CHILD_RVREF_FUNCTION(R, T, f) \
  R&& f() { return std::move<T>(*this).f(); }
#define CHILD_CONST_MEMBER(R, T, m) \
        R get_ ## m () const { return static_cast<T const&>(*this).m; }


namespace pyviennacl
{
  typedef std::vector<double>      cpu_vector_type;
  typedef viennacl::vector<double> vcl_vector_type;
  
  typedef vcl_vector_type::value_type value_type;
  typedef vcl_vector_type::size_type  size_type;
  
  template <class CRTP>
  class VCLVectorBase 
  {
  public:
    value_type operator[](size_type i) const
    {
      // Needs optimising
      return static_cast<CRTP const&>(*this)[i];
    }

    /*
    VCLVectorBase() {}

    VCLVectorBase(VCLVectorBase<CRTP> const& v)
      : VCLVectorBase()
    {
      this = &(VCLVectorBase<CRTP>)CRTP(v);
    }

    VCLVectorBase(VCLVectorBase<CRTP>&& v)
      : VCLVectorBase()
    {
      this = &(VCLVectorBase<CRTP>)CRTP(std::move<CRTP>(v));
    }
    */

    CHILD_FUNCTION(vcl_vector_type, CRTP, get_vector)
    CHILD_CONST_FUNCTION(CRTP, CRTP, get_lhs)
    CHILD_CONST_FUNCTION(CRTP, CRTP, get_rhs)
    CHILD_CONST_FUNCTION(size_type, CRTP, size)
    CHILD_CONST_FUNCTION(vcl_vector_type::const_iterator, CRTP, begin)
    CHILD_CONST_FUNCTION(vcl_vector_type::const_iterator, CRTP, end)

  };

  class VCLVector : public VCLVectorBase<VCLVector>
  {
    /*
    typedef typename VCLVectorBase<VCLVector>::cpu_vector_type cpu_vector_type;
    typedef typename VCLVectorBase<VCLVector>::vcl_vector_type vcl_vector_type;

    typedef typename VCLVectorBase<VCLVector>::size_type  size_type;
    typedef typename VCLVectorBase<VCLVector>::value_type value_type;
    */
   
    vcl_vector_type vector;

  public:

    // Need to reimplement using boost::numpy

    size_type size() const
    {
      return vector.size();
    }

    vcl_vector_type& get_vector()
    {
      std::cout << "VCLVectorL: " << vector.size() << std::endl;
      return std::forward<vcl_vector_type&>(vector);
    }

    vcl_vector_type::const_iterator begin() const
    {
      return vector.begin();
    }

    vcl_vector_type::const_iterator lend() const
    {
      return vector.end();
    }

    static void swap(VCLVector& a, VCLVector& b)
    {
      viennacl::fast_swap(a.get_vector(), b.get_vector());
    }

    VCLVector() {}

    VCLVector(size_type s)
    {
      vector = vcl_vector_type(s);
    }

    template <class CRTP>
    VCLVector(VCLVectorBase<CRTP> const& vcl_v)
      : VCLVector()
    {
      vector = vcl_vector_type(vcl_v.size());
      viennacl::copy(vcl_v.begin(), vcl_v.end(), vector.begin());
    }

    template <class CRTP>
    VCLVector(VCLVectorBase<CRTP>&& vcl_v)
      : VCLVector()
    {
      swap(*this, static_cast<VCLVector&>(vcl_v));
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

    VCLVector& operator=(VCLVector b)
    {
      swap(*this, b);
      return *this;
    }
      
  };

  template <class VCLVectorL, class VCLVectorR>
  class VCLVectorAdd : public VCLVectorBase<VCLVectorAdd<VCLVectorL, 
							 VCLVectorR> >
  {
    /*
    typedef typename VCLVectorBase<VCLVector>::cpu_vector_type cpu_vector_type;
    typedef typename VCLVectorBase<VCLVector>::vcl_vector_type vcl_vector_type;

    typedef typename VCLVectorBase<VCLVector>::size_type  size_type;
    typedef typename VCLVectorBase<VCLVector>::value_type value_type;
    */    

    vcl_vector_type vector;
    cpu_vector_type cpu_vector;

    VCLVectorL lhs;
    VCLVectorR rhs;

    int accessed = 0;
    int executed = 0;

  public:
    VCLVectorL& get_lhs() const
    {
      return std::forward<VCLVectorL>(lhs);
    }

    VCLVectorR& get_rhs() const
    {
      return std::forward<VCLVectorR>(rhs);
    }

    static void swap(VCLVectorBase<VCLVectorAdd<VCLVectorL, VCLVectorR> >& a,
		     VCLVectorBase<VCLVectorAdd<VCLVectorL, VCLVectorR> >& b)
    {
      VCLVectorL::swap(std::forward<VCLVectorL&>(a.get_lhs()), 
		       std::forward<VCLVectorL&>(b.get_lhs()));
      VCLVectorR::swap(std::forward<VCLVectorR&>(a.get_rhs()), 
		       std::forward<VCLVectorR&>(b.get_rhs()));
    }

    ~VCLVectorAdd() {}

    VCLVectorAdd() 
      : lhs(VCLVectorL()),
	rhs(VCLVectorR())
    { std::cout << "H\n"; }

    VCLVectorAdd(VCLVectorBase<VCLVectorL> const& _lhs,
		 VCLVectorBase<VCLVectorR> const& _rhs)
      : lhs((VCLVectorL)(_lhs)),
	rhs((VCLVectorR)(_rhs))
    {
      //only do the actual addition when we access the vector
      assert(_lhs.size() == _rhs.size());
      std::cout << "VCLVectorAdd: " << _lhs.size() 
		<< ", " << _rhs.size() << std::endl;
    }

    VCLVectorAdd(VCLVectorBase<VCLVectorAdd<VCLVectorL,VCLVectorR> > const& v)
      : VCLVectorAdd()
    {
      lhs = VCLVectorL(v.get_lhs());
      rhs = VCLVectorR(v.get_rhs());
    }

    VCLVectorAdd(VCLVectorBase<VCLVectorAdd<VCLVectorL, VCLVectorR> >&& v)
      : VCLVectorAdd()
    {
      swap(*this, v);
    }

    VCLVectorAdd<VCLVectorL, VCLVectorR>
    operator=(VCLVectorAdd<VCLVectorL, VCLVectorR> v)
    {
      swap(*this, v);
      return *this;
    }

    size_type size() const
    {
      if (!executed) {
	return 0;
      } else {
	return vector.size();
      }
      //vcl_vector_type const& v = get_vector();
      //return v.size();
    }

    vcl_vector_type get_vector()
    {
      std::cout << executed << std::endl;
      if (!executed) {
	this->vector = lhs.get_vector() + rhs.get_vector();
	executed = 1;
      }
      return vector;
    }

    /*
    // Need to reimplement using boost::numpy
    list get_value()
    {
      if (!accessed) {
	cpu_vector_type c(rhs.size());
	cpu_vector = c;
	viennacl::fast_copy(get_vector().begin(), get_vector().end(),
			    cpu_vector.begin());
      }
      list l;
      for (size_type i = 0; i < lhs.size(); ++i)
	l.append((double)cpu_vector[i]);
      return l;
    }

    value_type operator[](size_type i) const
    {
      if (accessed) {
	return cpu_vector[i];
      } else {
	cpu_vector.resize(lhs.size());
	viennacl::fast_copy(get_vector().begin(), get_vector().end(),
			    cpu_vector.begin());
	accessed = 1;
	return cpu_vector[i];
      }      
    }
    */
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

  template <class V>
  list vector_to_list(V v)
  {
    list l;
    vcl_vector_type vcl_vector(v.get_vector());
    size_type&& s = vcl_vector.size();
    cpu_vector_type cpu_vector(s);
    viennacl::fast_copy(vcl_vector.begin(), vcl_vector.end(),
			cpu_vector.begin());
    for (size_type i = 0; i < s; ++i)
      l.append((double)cpu_vector[i]);
    return l;
    }

}

BOOST_PYTHON_MODULE(_viennacl_extemp)
{
  using namespace pyviennacl;

  /*
  class_<VCLVectorAdd<VCLVectorAdd<VCLVectorAdd<VCLVector, VCLVector>, VCLVector>, VCLVector> >
    ("vector", no_init)
    .def(self + VCLVector())
    .def(VCLVector() + self)
    .def(self + self)
    .add_property("value", &vector_to_list<VCLVectorAdd<VCLVectorAdd<VCLVectorAdd<VCLVector, VCLVector>, VCLVector>, VCLVector> >)
    ;

  class_<VCLVectorAdd<VCLVectorAdd<VCLVector, VCLVector>, VCLVector> >
    ("vector", no_init)
    .def(self + VCLVector())
    .def(VCLVector() + self)
    .def(self + self)
    .add_property("value", &vector_to_list<VCLVectorAdd<VCLVectorAdd<VCLVector, VCLVector>, VCLVector> >)
    ;
  */

  class_<VCLVectorAdd<VCLVector, VCLVector> >
    ("vector", no_init)
    .def(self + VCLVector())
    //.def(VCLVector() + self)
    //.def(self + self)
    .add_property("value", &vector_to_list<VCLVectorAdd<VCLVector, VCLVector> >)
    ;
  
  class_<VCLVector>("vector")
    .def(init<int>())
    .def(init<list>())
    //.def(init<boost::python::object>())
    .def(self + self)
    //.def(self + VCLVectorAdd<VCLVector, VCLVector>())
    //.def(VCLVectorAdd<VCLVector, VCLVector>() + self)
    //.def(vector<viennacl::vector<double>, double>() + self)
    //.def(self += vector<viennacl::vector<double>, double>())
    .add_property("value", &vector_to_list<VCLVector>)
    ;

}

