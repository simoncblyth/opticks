#include <iostream>

#include "BRAP_API_EXPORT.hh"
#include "BRAP_FLAGS.hh"

template<typename T> 
BRAP_API void foo(T value)
{
    std::cerr << "BFoo"
              << " value " << value
              << std::endl 
              ;
}

template BRAP_API void foo<int>(int);
template BRAP_API void foo<double>(double);
template BRAP_API void foo<char*>(char*);


