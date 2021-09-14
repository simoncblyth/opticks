
#include <iostream>
#include <iomanip>
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


int main(int argc, char** argv)
{
    unsigned seed = 1u ; 
    //SRng<double> rng0(seed) ; 
    //test_rng<double>(rng0); 

    SRng<float> rng1(seed) ; 
    test_rng<float>(rng1); 

    return 0;
}



