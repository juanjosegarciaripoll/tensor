  //////////////////////////////////////////////////////////////////////
  // CREATE A SET OF INDICES COVERING i0:i1:i2 (MATLAB NOTATION)
  //

  static IndexRange *range2(index i0, index i2, index i1)
  {
    index l = (i2 - i0) / i1 + 1;
    Indices output(l);
    for (int i = 0; i < l; i++, i0 += i1) {
      output.at(i) = i0;
    }
    return new IndexRange(output);
  }
