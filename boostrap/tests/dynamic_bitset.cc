// boost- ; name=dynamic_bitset ; gcc $name.cc -std=c++11 -lstdc++ -I$(boost-prefix)/include -o /tmp/$name && /tmp/$name 

#include <iostream>
#include <boost/dynamic_bitset.hpp>

int main(int argc, char** argv)
{
    int nbit = argc > 1 ? std::atoi(argv[1]) : 16 ; 
    boost::dynamic_bitset<>* x = new boost::dynamic_bitset<>(nbit); 

    if( nbit == 100 ) 
    {
        for(int i=0 ; i < nbit ; ++i ) x->set(i)  ; 
    }
    else
    {
       for(int i=0 ; i < nbit ; ++i ) if(i % 7 == 0) x->set(i)  ; 
    }

    std::cout << *x << "\n";

    std::cout << " x->size() " << x->size() << std::endl ; 
    std::cout << " x->count() " << x->count() << std::endl ; 
    std::cout << " x->any() " << x->any() << std::endl ; 
    std::cout << " x->all() " << x->all() << std::endl ; 

    return 0 ; 
}
