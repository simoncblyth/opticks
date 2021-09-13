// name=rng ; gcc $name.cc -std=c++11 -lstdc++ -o /tmp/$name && /tmp/$name

// https://stackoverflow.com/questions/31417957/encapsulated-random-number-generator-in-c-11-using-boost

#include <iostream>
#include <iomanip>
#include <random>

template <typename T>
struct RNG
{
    std::mt19937_64 engine ;
    std::uniform_real_distribution<T>  dist ; 

    RNG(unsigned seed=0u); 
    T operator()() ; 
};


template<typename T> RNG<T>::RNG(unsigned seed_)
    :
    dist(0, 1)
{
    engine.seed(seed_); 
}

template<typename T> T RNG<T>::operator()()
{
    return dist(engine) ; 
}



void test_rng( const std::function<double()>& fn )
{
    double a ; 
    double b ; 
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


int main()
{
    unsigned seed = 1u ; 

    RNG<double> rng(seed) ; 

    test_rng(rng); 

    return 0;
}


