#include "BFoo.hh"


template <typename T>
void BCar::foo(T value)
{
    std::cerr << "BCar::foo"
              << " value " << value
              << std::endl 
              ;
}   


template BRAP_API void BCar::foo<int>(int);
template BRAP_API void BCar::foo<double>(double);
template BRAP_API void BCar::foo<char*>(char*);


