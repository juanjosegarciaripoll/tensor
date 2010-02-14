// -*- mode: c++; fill-column: 80; c-basic-offset: 2; indent-tabs-mode: nil -*-
//
// Copyright 2008, Juan Jose Garcia-Ripoll
//

#if !defined(TENSOR_TENSOR_H) || defined(TENSOR_DETAIL_TENSOR_SLICE_HPP)
#error "This header cannot be included manually"
#else
#define TENSOR_DETAIL_TENSOR_BASE_HPP

namespace tensor {

  template<typename elt_t>
  class Tensor<elt_t>::view
  {
  public:
    //typedef typename RefPointer<value_type>::elt_t elt_t;

    ~view();

    void set(elt_t v);
    elt_t ref() const;

    view &operator=(const view &a_stripe);
    view &operator=(const Tensor<elt_t> &a_tensor);
    view &operator=(elt_t v);

    operator Tensor<elt_t>() const;

  private:
    Vector<elt_t> data_;
    Indices dims_;
    Range *ranges_;

    // Start from another tensor and a set of ranges
    view(const Tensor<elt_t> &parent, Range *ranges) :
      data_(parent.data_), dims_(parent.dims_), ranges_(ranges)
    {}

    // We do not want these objects to be initialized by users nor copied.
    view();
    view(const view &a_tensor);
    view(const Tensor<elt_t> *a_tensor, ...);
    view(Tensor<elt_t> *a_tensor, ...);
  };

  typename<typename elt_t>
  Tensor<elt_t>::view Tensor<elt_t>::operator()(Range *r)
  {
    return view(*this, r);
  }

  typename<typename elt_t>
  Tensor<elt_t>::view Tensor<elt_t>::operator()(Range *r1, Range *r2)
  {
    Range *r = product(r1, r2);
    return view(*this, r);
  }

  typename<typename elt_t>
  Tensor<elt_t>::view Tensor<elt_t>::operator()(Range *r1, Range *r2, Range *r3)
  {
    Range *r = product(r1, product(r2, r3));
    return view(*this, r);
  }

  typename<typename elt_t>
  Tensor<elt_t>::view Tensor<elt_t>::operator()(Range *r1, Range *r2, Range *r3,
                                                Range *r4)
  {
    Range *r = product(r1, product(r2, product(r3, r4)));
    return view(*this, r);
  }

  typename<typename elt_t>
  Tensor<elt_t>::view Tensor<elt_t>::operator()(Range *r1, Range *r2, Range *r3,
                                                Range *r4, Range *r5)
  {
    Range *r = product(r1, product(r2, product(r3, product(r4, r5))));
    return view(*this, r);
  }

  typename<typename elt_t>
  Tensor<elt_t>::view Tensor<elt_t>::operator()(Range *r1, Range *r2, Range *r3,
                                                Range *r4, Range *r5, Range *r6)
  {
    Range *r = product(r1, product(r2, product(r3, product(r4, product(r5, r6)))));
    return view(*this, r);
  }

} // namespace tensor

#endif // !TENSOR_DETAIL_TENSOR_SLICE_HPP
