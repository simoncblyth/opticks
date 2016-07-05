#include <iostream>

#include "BRAP_API_EXPORT.hh"
#include "BRAP_HEAD.hh"

template<typename T> 
void foo(T value)
{
    std::cerr << "BFoo"
              << " value " << value
              << std::endl 
              ;
}

template BRAP_API void foo<int>(int);
template BRAP_API void foo<double>(double);
template BRAP_API void foo<char*>(char*);




////////////////  explicit instanciation in .hh /////////////

class BRAP_API BBar {
   public:
        template <typename T>
        void foo(T value)
        {
            std::cerr << "BBar::foo"
                      << " value " << value
                       << std::endl 
              ;

        }   
};


template BRAP_API void BBar::foo<int>(int);
template BRAP_API void BBar::foo<double>(double);
template BRAP_API void BBar::foo<char*>(char*);


////////////////  explicit instanciation in  .cc /////////////

class BRAP_API BCar {
   public:
        template <typename T>
        void foo(T value);

};

/////////////////////////////////////////////////////////////////////////



#include "BRAP_TAIL.hh"



