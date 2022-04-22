// name=SRngTest ; gcc $name.cc -std=c++11 -lstdc++ -I.. -o /tmp/$name && /tmp/$name
#include <iostream>
#include <iomanip>
#include <functional>
#include "SRng.hh"

template <typename T> void test_rng( const std::function<T()>& fn )
{
    T a ; 
    T b ; 
    bool done ; 
    unsigned count = 0 ; 

    do {
        a = fn() ; 
        b = fn() ; 

        std::cout 
            << " count " << std::setw(10) <<  count 
            << " a " << std::fixed << std::setw(10) << std::setprecision(4) <<  a 
            << " b " << std::fixed << std::setw(10) << std::setprecision(4) <<  b 
            << std::endl
            ; 

        done = a > 0.99 && b > 0.99 ; 
        count += 1 ;   

    } while( done == false ) ; 


    std::cout 
        << " result " 
        << " count " << count 
        << " a " << a 
        << " b " << b 
        << std::endl
        ; 
}


void test_rng_0()
{
    unsigned seed = 1u ; 
    SRng<double> rng0(seed) ; 
    test_rng<double>(rng0); 
}

void test_rng_1()
{
    unsigned seed = 1u ; 
    SRng<float> rng1(seed) ; 
    test_rng<float>(rng1); 
}


int main(int argc, char** argv)
{
    //test_rng_0();  
    test_rng_1();  
    return 0;
}



