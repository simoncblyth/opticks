// name=do_while_continue ; gcc $name.cc -std=c++11 -lstdc++ -o /tmp/$name && /tmp/$name
/**
https://stackoverflow.com/questions/9878965/rand-between-0-and-1
**/
#include <iostream>
#include <random>
#include <chrono>

bool do_while_condition(double a, double b)
{
    std::cout << "do_while_condition a " << a << " b " << b  << std::endl ; 
    return true ; 
}

int main()
{
    std::mt19937_64 rng;
    uint64_t timeSeed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::seed_seq ss{uint32_t(timeSeed & 0xffffffff), uint32_t(timeSeed>>32)};
    rng.seed(ss);
    std::uniform_real_distribution<double> unif(0, 1);

    bool continue_bug = true ; 

    double a ; 
    double b ; 

    do {
        a = unif(rng);
        std::cout << "head a " << a << std::endl;
     
        if( a < 0.5 && continue_bug  )
        {
            std::cout << "continue " << std::endl ; 
            continue ;   
            // GOTO WHILE CONDITION POTENTIALLY USING b UNINITIALIZED 
            // SO IN GENERAL WHAT THIS CODE DOES IS UNDEFINED
        }

        b = unif(rng); 
        std::cout << "tail b " << b << std::endl ; 

    } while( ( a < 0.9 || b < 0.9 ) && do_while_condition(a,b) ) ; 

    // looping condition 

    std::cout 
        << " result " 
        << " a " << a 
        << " b " << b 
        << std::endl
        ; 

    return 0;
}


