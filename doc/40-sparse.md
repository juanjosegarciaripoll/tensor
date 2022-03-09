# Sparse matrices {#sparse}

The classes RSparse and CSparse implement our library's sparse matrices class. These are tensors with two indices in which most of the entries are zero, what allows a more efficient storage and manipulation of the matrix entries.

Sparse matrices can be created from a tensor, extracting only the nonzero components
```{.cc}
Tensor<double> I = Tensor<double>::eye(50);
Sparse<double> S(I);
```
but there are also more efficient means of creating these matrices with specialized functions for diagonal, random, and tensor products.
```{.cc}
Sparse<double> S = Sparse<double>::eye(50,50);
Sparse<double> A = kron(S, S);
```
