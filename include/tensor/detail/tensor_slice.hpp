// -*- mode: c++; fill-column: 80; c-basic-offset: 2; indent-tabs-mode: nil -*-
//
// Copyright 2008, Juan Jose Garcia-Ripoll
//

#if !defined(TENSOR_TENSOR_H) || defined(TENSOR_DETAIL_TENSOR_SLICE_HPP)
#error "This header cannot be included manually"
#else
#define TENSOR_DETAIL_TENSOR_SLICE_HPP

namespace tensor {

  template<typename elt_t>
  class Tensor<elt_t>::view
  {
  public:
    ~view() {
      delete ranges_;
    };
    operator Tensor<elt_t>() const;

  private:
    const Vector<elt_t> data_;
    Indices dims_;
    Range *ranges_;

    // Start from another tensor and a set of ranges
    view(const Tensor<elt_t> &parent, Indices &dims, Range *ranges) :
      data_(parent.data_), dims_(dims), ranges_(ranges)
    {}

    // We do not want these objects to be initialized by users nor copied.
    view();
    view(const view &a_tensor);
    view(const Tensor<elt_t> *a_tensor, ...);
    view(Tensor<elt_t> *a_tensor, ...);

    friend class Tensor<elt_t>;
  };

  template<typename elt_t>
  class Tensor<elt_t>::mutable_view
  {
  public:
    ~mutable_view() {
      delete ranges_;
    };

    void set(elt_t v);
    elt_t ref() const;

    mutable_view &operator=(const mutable_view &a_stripe);
    mutable_view &operator=(const Tensor<elt_t> &a_tensor);
    mutable_view &operator=(elt_t v);

    operator Tensor<elt_t>() const;

  private:
    VectorView<elt_t> data_;
    Indices dims_;
    Range *ranges_;

    // Start from another tensor and a set of ranges
    mutable_view(const Tensor<elt_t> &parent, Indices &dims, Range *ranges) :
      data_(parent.data_), dims_(dims), ranges_(ranges)
    {}

    // We do not want these objects to be initialized by users nor copied.
    mutable_view();
    mutable_view(const mutable_view &a_tensor);
    mutable_view(const Tensor<elt_t> *a_tensor, ...);
    mutable_view(Tensor<elt_t> *a_tensor, ...);

    friend class Tensor<elt_t>;
  };

  Range *product(Range *r1, Range *r2);

} // namespace tensor

#endif // !TENSOR_DETAIL_TENSOR_SLICE_HPP
