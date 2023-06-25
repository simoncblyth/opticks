// name=populate_array_test ; gcc $name.cc -std=c++11 -lstdc++ -o /tmp/$name && /tmp/$name

#include <iostream>
#include <sstream>
#include <string>

const int N = 16 ; 

template<typename T>
std::string desc( const T* tt, int num )
{
    std::stringstream ss ; 
    for(int i=0 ; i < num ; i++ ) ss << tt[i] << " " ; 
    ss << std::endl ; 
    std::string str = ss.str(); 
    return str ;  
}

/**
populate_array_0
------------------

pointer-to-pointer argument enables the function 
to populate a structure from the calling scope

**/

template<typename T>
void populate_array_0( T** arr, int num ) 
{
    for(int i=0 ; i < num ; i++) (*arr)[i] = T(i) ; 
}
void test_0()
{
    float ss[N] ;
    float* ss_ptr = ss ; 
    populate_array_0( &ss_ptr, N ); 

    std::cout << "test_0: " << desc(ss, N) ; 
}

/**
populate_array_1
-------------------

BUT : no need for pointer-to-pointer
because there us no need to change the location
of that structure. Just need to fill it, so 
one level of indirection is fine. 

**/

template<typename T>
void populate_array_1( T* arr, int num ) 
{
    for(int i=0 ; i < num ; i++) arr[i] = T(i) ; 
}
void test_1()
{
    float ss[N] ;
    populate_array_1( ss, N ); 
    std::cout << "test_1: " << desc(ss, N) ; 
}


int main(int argc, char** argv)
{
    test_0(); 
    test_1(); 
    return 0 ; 
}
