#include <iostream>

#define CHILD_CONST_FUNCTION(R, T, f) \
        R f() const { return static_cast<T const&>(*this).f(); }
#define CHILD_FUNCTION(R, T, f) \
  R& f() { return std::forward<T&>(*this).f(); }
#define CHILD_RVREF_FUNCTION(R, T, f) \
  R&& f() { return std::move<T>(*this).f(); }
#define CHILD_CONST_MEMBER(R, T, m) \
        R get_ ## m () const { return static_cast<T const&>(*this).m; }


template <class T>
class vector_expession
{

  CHILD_FUNCTION(T,T, result);
  
};

class UnaryVector
{
  int value;

public:
  ~UnaryVector() {}

  UnaryVector() : value(1) {}

  UnaryVector(UnaryVector const& v)
    : UnaryVector()
  {
    value = v.value;
  }

  UnaryVector(UnaryVector&& v)
    : UnaryVector()
  {
    value = v.value;
    v.value = 0;
  }

  UnaryVector& operator=(UnaryVector v)
  {
    value = v.value;
    v.value = 0;
  }

  UnaryVector& result()
  {
    return (*this);
  }

  int get_value()
  {
    return value;
  }

};

template <class L, class R>
class SumVector : public Vector<SumVector<L, R> >
{
  L l;
  R r;

public:
  ~SumVector() {}

  SumVector() : l(L()), r(R()) {}

  SumVector(Vector<L> const& v_l, Vector<R> const& v_r)
    : l(v_l), r(v_r)
  {
    //assert();
  }

  SumVector(Vector<L>&& v_l, Vector<R>&& r)
    : SumVector()
  {
    l = v_l;
    r = v_r;

    v_l = NULL; v_r = NULL;
  }

  UnaryVector& operator=(UnaryVector v)
  {
    value = v.value;
    v.value = 0;
  }

  UnaryVector& result()
  {
    return (*this);
  }

  int get_value()
  {
    return value;
  }
};
