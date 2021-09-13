// name=random ; gcc $name.cc -std=c++11 -lstdc++ -o /tmp/$name && /tmp/$name
/**
https://stackoverflow.com/questions/9878965/rand-between-0-and-1

https://stackoverflow.com/questions/31417957/encapsulated-random-number-generator-in-c-11-using-boost

**/
#include <iostream>
#include <iomanip>
#include <random>



int main()
{
    std::mt19937_64 rng;

    unsigned seed = 0u ; 
    rng.seed(seed);
    std::uniform_real_distribution<double> unif(0, 1);

    double a ; 
    double b ; 
    bool done ; 
    unsigned count = 0 ; 

    do {
        a = unif(rng);
        b = unif(rng);
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

    return 0;
}


