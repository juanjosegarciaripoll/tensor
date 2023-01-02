#pragma once

#include <memory>
#include <iostream>
#include <tensor/exceptions.h>

namespace tensor {

#ifdef TENSOR_SHARED_ARRAY_PTR

template <typename elt>
class shared_array {
 public:
  shared_array() = default;
  shared_array(const shared_array &other)
      : pointer_{other.pointer_}, counter_{other.get_counter()} {}
  shared_array(shared_array &&other) { this->swap(other); }
  shared_array &operator=(const shared_array &other) {
    if tensor_likely (&other != this) {
      delete_pointer();
      counter_ = other.get_counter();
      pointer_ = other.pointer_;
    }
    return *this;
  }
  shared_array &operator=(shared_array &&other) {
    delete_pointer();
    this->swap(other);
    return *this;
  }
  explicit shared_array(elt *pointer, bool owned = false)
      : pointer_{pointer},
        counter_{owned ? nullptr : new counter_structure{2}} {}
  ~shared_array() { delete_pointer(); }

  elt &operator*() const noexcept { return *pointer_; }

  elt *operator->() const noexcept { return pointer_; }

  elt *get() const noexcept { return pointer_; }

  void swap(shared_array &other) noexcept {
    std::swap(counter_, other.counter_);
    std::swap(pointer_, other.pointer_);
  }

  void reset(elt *other) {
    delete_pointer();
    pointer_ = other;
  }

  long use_count() const noexcept {
    if (counter_ == nullptr) {
      return 1;
    } else {
      return counter_->use_count;
    }
  }

  bool shared() const noexcept {
    return counter_ != nullptr && (counter_->use_count > 1);
  }

  bool unique() const noexcept {
    return counter_ == nullptr || (counter_->use_count == 1);
  }

 private:
  struct counter_structure {
    long use_count{2};
  };

  void delete_pointer() {
    if tensor_unlikely (counter_ != nullptr) {
      complex_deletion();
    } else {
      delete[] pointer_;
      pointer_ = nullptr;
    }
  }

  void complex_deletion() {
    if (--(counter_->use_count) <= 0) {
      delete counter_;
      delete[] pointer_;
    }
    counter_ = nullptr;
    pointer_ = nullptr;
  }

  counter_structure *get_counter() const {
    if tensor_likely (counter_ == nullptr) {
      return counter_ = new counter_structure();
    }
    ++(counter_->use_count);
    return counter_;
  };
  using counter_ptr_t = counter_structure *;

  elt *pointer_{nullptr};
  mutable counter_ptr_t counter_{nullptr};
};

template <class T>
inline shared_array<T> make_shared_array(size_t size) {
  return shared_array<T>(new T[size]);
}

template <class T>
inline shared_array<T> make_shared_array_from_ptr(T *data) {
  return shared_array<T>(data, true);
}

#else  // !TENSOR_SHARED_PTR

template <typename T>
using shared_array = std::shared_ptr<T[]>;

template <typename T>
inline shared_array<T> make_shared_array(size_t size) {
  return shared_array<T>(new T[size], std::default_delete<T[]>());
}

template <typename T>
inline shared_array<T> make_shared_array_from_ptr(T *data) {
  return shared_array<T>(data, [](T *) {});
}

#endif

}  // namespace tensor
