#!/bin/sh
(find src include tests -type f -name "*.cc" -print0; find src/sparse include tests -type f -name "*.h" -print0; find src include tests -type f -name "*.hpp" -print0) | xargs -0 clang-format -style=file -i
