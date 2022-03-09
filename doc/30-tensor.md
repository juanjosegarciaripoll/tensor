# Tensor objects {#tensor}

## Tensor class {#tensor_class}

In programming jargon, a `Tensor` is a multidimensional array of numbers. Mathematically, a tensor is an association between collections of integers, the *indices*, and the values that are stored in memory. Tensors can have any number of indices, but tensors with one index are usually called *vectors* and tensors with two indices are called *matrices*.

### Class invariants

A tensor is formed essentially by three elements:
- A `Vector` that contains all the values associated to the tensor in dynamically allocated memory.
- The `Dimensions` of the tensor, which define the number of indices (the *rank*) and the limits of those indices (the *dimension size*).
- The mapping between tensor indices and positions into the tensor's data. In this library, the mapplin gis [column-major order](#columnmajor), as explained before.

In this library, a tensor `T` is an object with certain invariants:
- The dimensions of the tensor `T.dimensions()` is an immutable object formed by non-negative integers.
- The number of dimensions is the `T.rank()`. It can be any non-negative number. In practice, it is a small number, between 1 and 6, and never above 32 because of memory limitations.
- The size of the tensor `T.size()` is the product of the tensor's dimensions.
- The data of the tensor may be [accessed](#tensor_access) using a number of indices `T(i1,...,in)`, `T.at(i1,...,in)`, that is equal to the rank of the tensor.
- The tensor's vector of values can be access sequentially using a single number, `T[i]`.
- Tensors can be copied. They have C++ value semantics and also C++11 move semantics.
- Tensors are templated objects. They can be specialized to different number types, although the library only provides specializations for real and complex double-precision numbers.

Some examples of how tensors are created and used:
```{.cc}
  Tensor<double> a = Tensor<double>::eye(2,2); // creates a 2x2 identity matrix
  double one = a(0,0);                         // returns the upper left element
  Tensor<double> b = a(range(0), _);           // returns first row of the matrix
  Tensor<double> c = reshape(b, 1, 4);         // converts b into a row vector
  RTensor d = reshape(b, 1, 4);                // same as above, RTensor is an alias
```

### Dimensions {#tensor_dimensions}

A `Dimensions` object `d` is an immutable instance that stores the dimensions of a tensor. The object is formed by a fixed number, the *rank* `d.rank()`, of non-negative integers. The size of a dimensions object is, in principle, unlimited. In practice, we cannot create tensor objects with more than 32 indices, because of memory constraints. Consider that in the simplest case in which each dimension has size 2, an object with `2**32` elements exceeds the memory size of most computers.

Dimensions are immutable by design, because we cannot change the shape of a tensor. However, dimensions object can be constructed using either braced initialization, or a vector of integers of type `Indices` (see [Indices](#Indices)), as shown below
```{.cc}
Dimensions d {2, 3, 4};
RTensor A(d);             // Create a 2 by 3 by 4 tensor with 24 elements
Indices i {2, 3, 4};
RTensor A(Dimensions(i)); // Same as above
```

### Constructors {#tensor_creation}

To create a new tensor with uninitialized content, you can use a constructor that takes a `Dimensions` list or use the `empty()` static functions that takes a series of non-negative integers as dimension sizes:
```{.cc}
CTensor ca = CTensor(Dimensions{2, 3});    // creates a 2x3 matrix of complex doubles
RTensor ra = RTensor(Dimensions{5, 4, 3)}; // creates a 5x4x3 tensor of doubles
CTensor rb = RTensor::empty(5, 4);         // creates a 5x4 tensor
```

For convenience, there exist other static functions that produce initialized tensors
```{.cc}
RTensor a = RTensor::random(3, 2);  // fill the tensor with random content
CTensor b = CTensor::eye(5, 5);     // identity matrix
CTensor c = CTensor::zeros(3, 4, 5);// fills the tensor with zeros
RTensor d = RTensor::ones(4, 3);    // fills the tensor with ones
```

Tensors can also be initialized using C++ braced initializer lists. The lists must either contain elements of the same type, or other lists with equal lengths. Initializer lists can be nested up to 4-th order, as shown below:
```{.cc}
// Vector with 3 elements
RTensor a = {1.0, 2.0, 3.0};
// 2 by 3 matrix. We list 2 rows of 3 elements
RTensor b = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
```
All initializer lists must have matching dimensions. Unfortunately, because we use C++14, this can only be detected at run time.

### Iteration {#tensor_iteration}

Tensors behave very much like other C++ containers, such as [std::vector](https://en.cppreference.com/w/cpp/container/vector), allowing the use of iterations and algorithms. For instance, the following code changes the sign of a tensor:
```{.cc}
#include <tensor/tensor.h>

RTensor negate(const RTensor &a) {
  RTensor output = a.clone();
  for (auto &x : output) {
    x = -x;
  }
  return output;
}
```

The same algorith, written using the STL [std::transform](https://en.cppreference.com/w/cpp/algorithm/transform) algorithm would be
```{.cc}
#include <algorithm>
#include <functional>
#include <tensor/tensor.h>

RTensor negate(const RTensor &a) {
  RTensor output = a.clone();
  std::transform(std::begin(output), std::end(output),
                 std::begin(output), std::negate);
  return output;
}
```

@note Of course, a much simpler version is to write `-a` using the fact that our library already provides an `operator-()` for tensor objects, but these are examples to illustrate the use of iterators.

## Accessing elements {#tensor_access}

### Tensor indices {#tensor_index}

Tensor indices have the type `tensor::index`. They are positive or negative integers in the range `[-L,L-1]`, where `L` is the size of the corresponding dimension.
- Non-negative elements `0`, `1`, ... `L-1` are canonical index values.
- Negative elements `-L`, `-L+1`,... `-1` are wrapped indices that count the distance from the last value of the given index.

Thus, given a vector `V` with `L=4` elements:
- `V(0)` is the first element of the vector. 
- `V(3)` is the last element of the vector.
- `V(-1)` denotes the same position as `V(3)`.
- `V(-4)` denotes the same position as `V(0)`.
- `V(-5)` or `V(4)` are erroneous calls that result in [out of bound errors](#errors).

### Tensor ranges {#tensor_ranges}

One or more indices can be grouped into a *range*. Such ranges can be used to access a whole section of a vector, matrix or a tensor, implementing relatively sophisticated operations on the data.

Ranges are created using the `range` function, which can take different types of arguments:
- `range()` or `_` refers to all values of the given dimension.
- `range(n)` selects just the value `n`
- `range(a,b)` takes all values from `a` up to possibly including`b`.
- `range(a,b,s)` takes all values `a`, `a+s`, `a+2*s`, up to the index denoted by `b`.
- `range(ndx)` where `ndx` is a vector of type `Indices` with the allowed indices ([see below](#Indices)).

Ranges follow the same conventions as indices, whereby negative values are interpreted relative to the dimension of the object they refer to. For example, `range(0,-1)` denotes a full range, that covers all values that an index can take. In the following example, a range is created to denote a full range, and it is later on interpreted as such when used to [slice a tensor](#tensor_slices)
```{.cc}
auto r = range(0, -1); // Incomplete object denoting a full range
RTensor a = RTensor::random(3, 4);
RTensor b = a(r, range(0)); // Object r completed to produce a single column
```

@warning Note that `range(0, 3)` includes the last element `{0,1,2,3}`, unlike in Python, where it means `[0,1,2]`. This may change incompatibly in future releases.

### Element-wise access {#tensor_element_access}

Given one or more indices, we can access a single element of a tensor in different ways:
- We can get a const or mutable reference to the tensor value. A const reference provides us with a read-only access to the tensor, while a mutable reference provides us with the possibility of changing the tensor's value.
- We can address the elements of the tensor sequentially, as if it were a vector, irrespective of the rank, or we can use a collection of indices.

This results in three possible calls:
1. A const reference to the elements of a tensor in column major oder, using `operator[]` and a single index limited by the size of the tensor:
```{.cc}
RTensor t = RTensor::random(3, 4);
for (index i = 0; i < 12; ++i) {
  std::cout << "t[" << i >> "]=" << t[i] << '\n';
}
```
2. A const reference to the elements accessed using the indices in parenthesis, as `t(i,j,k)`. The number of indices must be equal to the rank of the tensor and they can take positive or negative values, limited by the respective dimensions:
```{.cc}
RTensor t = RTensor::random(3, 4);
double a_first = t(0, 0); // first element
double b_last  = t(2, 3); // last element
double c_err = t(3, 4); // out of bounds error
double d_last = t(-1, 3); // last element (equal to b_last)
double e_last = t(2, -1); // same as above
```
3. A mutable reference to the tensor is obtained using the `at()` method, as in `t.at(i,j,k)`
```{.cc}
RTensor t = RTensor::random(3, 4);
t.at(0, 0) = 0;  // first element
t.at(2, 3) = 11; // last element
t.at(-2,-1) = 10; // similar to modifying t.at(1,3)
t.at(3, 4) = 3;  // out of bounds error
const RTensor a = RTensor::random(10,2)
a.at(3, 2) = 3.0; // compiler error: we cannot use at() on an immutable tensor
```

@warning Currently `operator()` always returns const references, regardless of the object's const class. This may change in the future.

### Tensor slicing {#tensor_slices}

We can use the `operator()` and `at()` method to access more than one element in a tensor using [ranges](#ranges). As with element-wise access, the `operator()` returns an immutable view into the tensor's values with a new rank and shape given by the ranges, while `at()` provides a mutable view that can be changed.

The following example shows the use of immutable views:
```{.cc}
CTensor t = CTensor::random(5,5);

CTensor full  = t(_, _);                // no argument  -> take all Indices
CTensor row   = t(range(2), _);         // one argument -> take only given index
CTensor full2 = t(range(1,-1), _);      // two arguments-> start and stop
CTensor evenRows = t(range(1,-1,2), _); // three args   -> start,stop,stride
CTensor oddRows = t(range({1,2,3}), _); // Indices as argument
```

These examples use mutable views to modify th econtent of a non-const tensor:
```{.cc}
CTensor dest = CTensor::zeros(5,3);
CTensor col  = CTensor::ones(5,1);

dest.at(_, range(0)) = 5.0;     // first column gets complex value (5.0, 0)
dest.at(_, range(1)) = col;     // second column gets content of col
dest.at(_, range(2)) = dest(_, range(1));  // copy to third column
```

Views are private objects that should not be passed to other functions, returned from a function or stored in a variable. The reason is that the lifetime of a view cannot exceed the lifetime of the object it refers to. Fortunately, the library is intelligent enough to convert those views into fresh new copies of the tensor with its own lifetime guarantee. In the following example, `view` is actually of type `CTensor` and is a new object that contains two columns from the tensor `t`:
```{.cc}
CTensor t = CTensor::random(5,5);
auto view = t(_, range(0,1));  // Creates a new tensor
```

## Tensor shapes and ranks {#tensor_shape}

To query the dimensions, the tensor class offers a couple of functions:
```{.cc}
CTensor t = CTensor::zeros(5, 4, 3);

index rank = t.rank();           // number of dimensions: 3
index size = t.size();           // total size: 5*4*3 = 60

int d1 = t::dimension(0);        // number of entries in first dimension: 5
int d2 = t::dimension(1);        // along second dimension: 4
int d3 = t::dimension(2);        // along third dimension: 3

t::get_dimensions(&d1, &d2, &d3);// same: returns the three dimensions

Indices dims = t::dimensions();  // Returns an index vector with the dimensions
d1 = dims[0];                    // etc.
```

The function dimensions() is handy to produce an Indices vector that can later be used for convenient manipulations if you want to abstract away the exact rank of the tensor.
For two-dimensional tensors, and only for those, the class also offers the rows() and columns() functions, which are equivalent to dimension(0) and dimension(1);

If you want to modify the shape of a tensor, you can use the reshape() function
```{.cc}
CTensor t = CTensor::random(5,4);
CTensor t2 = reshape(t, 4, 5);

t.dimension(0) == t2.dimension(1);  // both comparisons evaluate to true
t.dimension(1) == t2.dimension(0);
```

## Vectors {#vectors}

### Tensor data

Internally, tensors use a templated object of class `Vector` to store the values that form a tensor. This object should not be used directly, as its implementation is likely to change in the future.

### Vectors of bool {#Booleans}

The library provides a vector Boolean values taking `true` or `false`. Such vectors can be produced when comparing tensors:
```{.cc}
RTensor a = {{1.0, 2.0}, {3.0, 4.0}};
RTensor b = {{1.0, 0.0}, {0.0, 1.0}};
Boolean x = (a == b);
```

### Vectors of indices {#Indices}

We also provide a stable object called `Indices` that stores a collection of integers used for indexing, slicing and reshaping tensor objects. At its heart, an Indices object is just a fixed-sized vector of integer values.
It uses the same [memory model](#tensor_sharing) as the tensor class, providing `operator()` and `at()` member functions to read and write the vector's values.

Indices can be constructed either by braced initialization or using static functions. The first case is the most recommended one
```{.cc}
Indices ndx = {1, 3, 5, 7}; \\ Odd indices
RTensor a = RTensor::random(8);
return a(range(ndx));  \\ Access odd indices from tensor
```

It is also possible to start from an uninitialized vector of fixed size and assign the elements according to some algorithm:
```{.cc}
Indices r = Indices::empty(3);
index i = 0;
r.at(0) = i++;
r.at(1) = i++;
r.at(2) = i++;
...
```
For convenience, use the function `iota()` to create vectors of equispaced integers:
```{.cc}
Indices r = iota(2, 10, 3); // {2, 5, 8}
```

The functions all_equal() and some_unequal() can be used to compare Indices. Furthermore, all comparison operators (==, <= etc.) are overloaded, and will return a vector of booleans that gives the elementwise result of the comparison.
```{.cc}
Indices i1 {1, 2, 3};
Indices i2 {1, 2, 5};

bool eq = all_equal(i1, i2);      // evaluates to false
Boolean comp = (i1 == i2)         // Boolean{true, true, false}
```


[Dimensions] @ref class Dimensions
