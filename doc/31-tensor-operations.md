# Tensor operations {#tensor_operations}

The following discussion of operations is not intented to be complete, and several of the more arcane or complex functions are not detailed here. For these, see either the library reference or have a look in the include files (especially tensor.h). All functions, like the whole rest of the code, reside in the tensor namespace. Consequently, you either have to prepend all functions here with "tensor::", or import them or the whole tensor namespace with a using-directive.

## Structural changes {#tensor_reshape}

The dimensions of a tensor cannot be changed and a tensor cannot be resized to have more or less elements than it already contains. However, we can create new objects that contain the same data with different dimensions.

The first structural change is the reshaping of a tensor, which means using the same [column-major ordered data](#columnmajor) with different dimensions. Similar to Matlab's or Numpy's counterparts, the functions `reshape()` changes explicitly the dimensions of a tensor, providing new ones
```{.cc}
CTensor a = CTensor::random(5, 4);
CTensor b = reshape(a, 1, 2, 2, 5); // b is a 1x2x2x5 tensor with the same data as a
```
The function `squeeze()` is slightly different, in that it removes dimensions that carry no information, because they have size one:
```{.cc}
CTensor c = squeeze(b);             // c is now a 2x2x5 tensor
CTensor d = permute(a, 0, -1);      // swaps first and last index, d(i,j) == a(j,i)
```

Another structural change is to modify the order of the indices that we use to access a tensor. For instance, we may want to permute the indices of a three-dimensional tensor, swapping the first and the last one
```{.cc}
CTensor a = CTensor::random(5, 3, 4); 
CTensor b = permute(a, 0, -1);      // swaps first and last index
a(0,2,3) == b(3,2,0) // returns always true
```

## Unary operations {#tensor_unop}

The library contains a limited number of unary elementwise operators for tensors. These include:
1. Element-wise negation `(-A)`.
2. Extracting real and imaginary parts, `real(A)` or `imag(A)`, or absolute values `abs(A)`.
3. Computing special functions, such as `cos()`, `sin()`, `exp()`, etc.

Out of these, only the first two are defined generically. The special functions are only defined for tensors of type `RTensor` and `CTensor`.

Some examples of these operations
```{.cc}
// Create a complex number with modulus 0.5 and random phase
RTensor re = RTensor::random(2, 3, 4);
RTensor im = RTensor::random(r.dimensions());
CTensor z1 = (cos(re) - 1_i * im) / 2.0;

// Mathematically equivalent to the code above. Note how we
// let the compiler infer the complex type
auto z2 = exp(RTensor::random(2, 3, 4)) / 2.0;
```

## Binary operations {#tensor_binop}

Binary operators are those that involve exactly two tensors, or a tensor and a number. These include
1. Element-wise addition, subtraction, multiplication and division of tensors, or tensors with numbers.
2. Element-wise comparison in column-major order, such as `A==B`, `A<=B`, which return a vector of `Boolean`.
3. Identity operators `all_equal()` and `some_unequal()`, which return `true` or `false`.

All of these operations are defined generically and work for all types of tensors, not just the library-supported `RTensor` and `CTensor`.

## Tensor contraction, scaling and tracing {#tensor_fold}

### Contraction

Tensor contraction is a generalization of matrix multiplication for tensor with more than two indices and come in two flavors.

The first flavor is "exterior" contraction, in which the indices of the tensor are simply joined together.
This is done with the function `fold()`, which takes four arguments, as in `fold(A, nA, B, nB)`.
The two arguments A and B are two tensors, while nA and nB represent the indices that are contracted.

For instance, assuming that A has two and B three indices, the code @c C=fold(A,0,B,1) contracts the second and first index of tensors A and B, respectively, as given by the formula
@f[
C_{a_1 b_0 b_2} = \sum_j A_{j a_1} B_{b_0 j b_2}
@f]
Note that the indices of B are simply added to those of A, in that precise order.

The other flavor is "internal" contraction, in which indices are replaced. Going back to the previous example, @c C=foldin(A,0,B,1) contracts tensors A and B according to the formula
@f[
C_{b_0 a_1 b_2} = \sum_j A_{j a_1} B_{b_0 j b_2}
@f]
Now the uncontracted indices of A appear in the place of the contracted index of B. This routine is very useful for applying transformations on certain indices of a tensor. Typically, A is a matrix describing an operator that only modifies the index / degree of freedom j.

Various derivates of these functions exist. The routine `foldc()` uses the complex conjugate of A for the contraction.
The routines `fold_into()` and `foldin_into()` allow you to specify the target tensor as the first element.
This is useful to avoid expensive memory allocation if you already have a fitting tensor for C lying around.

### Scaling

Scaling is a case of Hadamard multiplication, where the values of two tensors that share some indices are multiplied whenever those indices coincide. This library only supports scaling a tensor by a vector, implemented by the functions `scale()` and `scale_inplace()`.

Assuming a three-dimensional tensor `A` and a one-dimensional tensor (vector) V, the expression @c C=scale(A,2,V) scales A according to the formula
@f[
C_{a_0 a_1 a_2} = A_{a_0 a_1 a_2} V_{a_2}
@f]
A variant `scale_inplace()` exists that stores the result of the scaling directly in A.

### Tracing

Tracing is the operation of summing two indices from the same tensor, to create a tensor with less indices. This is implemented by the function `trace()`. For a tensor `A` with three indices, the function @c C=trace(A,0,2) calculates C as
@f[
C_{k} = \sum_i A_{i k i}
@f]

### Tensor reductions

Various functions allow the reduction of one or two tensors to a single value.
The functions `sum()` and `mean()` return the expected sum or mean value of all tensor entries.

The scalar product @c scprod(A,B) of two tensors A, B (interpreted as one-dimensional vectors of numbers) is given by
@f[
\sum_i A_i B_i^\ast
@f]

Two norms are provided: A maximum norm @c norm0(A) returns the entry of A with the largest absolute value.
The usual L2-norm, @c norm2(A) is equivalent to the expression @c sqrt(scprod(A,A)).


## Matrix operations

Even though matrices are just a special case of a tensor, they are common enough that some abstractions have their own names and are implemented as convenience functions:

1. The multiplication of two matrices `A` and `B` is a special case of tensor contraction, implemented by the function `mmult(A,B)`, wich is equivalent to `fold(A, -1, B, 0)`.
2. It is possible to constrauct a diagonal matrix using the function `diag()`. For instance, @c A=diag(V,-1,5,6) returns a 5x6 matrix that is zero except for the first side diagonal, which contains the elements of V, i.e., \f$ A_{j, j-1} = V_{j} \f$).
3. Similarly, one can extract the k-th diagonal of a matrix using `take_diag(A, k)`.
4. The function `transpose(A)` permutes the first and the last index of a tensor, implementing matrix transposition.
5. The function `adjoint(A)` transposes and takes the complex conjugate of a matrix.
6. The `trace()` of a matrix is the sum of its diagonal elements.

