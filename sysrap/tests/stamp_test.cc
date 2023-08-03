// ./stamp_test.sh 



#include <chrono>
#include <iostream>
#include <iomanip>

#include "stamp.h"
#include "NP.hh"
#include "s_time.h"

/**
test_performance
----------------

Old laptop: counting and stamp recording to 25M takes roughly 1s (1M us)
so 25 stamp records takes ~1us 
 
**/

void test_performance()
{
    static const int N = 1000000*25 ;  

    NP* t = NP::Make<uint64_t>(N) ;  
    uint64_t* tt = t->values<uint64_t>(); 
    for(int i=0 ; i < N ; i++) tt[i] = stamp::Now(); 
    t->save("$TTPATH");  
}

void test_reverse()
{
    uint64_t t0 = stamp::Now(); 

    using Clock = std::chrono::system_clock;
    using Unit  = std::chrono::microseconds  ; 

    std::chrono::time_point<Clock> t0_{Unit{t0}} ; 
    uint64_t t1 = std::chrono::duration_cast<Unit>(t0_.time_since_epoch()).count() ; 

    std::cout << "t0: " << t0 <<  std::endl ; 
    std::cout << "t1: " << t1 <<  std::endl ; 

    assert( t0 == t1 ) ; 
}

void test_format()
{
    uint64_t t0 = stamp::Now(); 
    std::cout << t0 << " : " << stamp::Format(t0) << " " << stamp::Format() << std::endl ; 

    std::cout << "stamp::Format()             " << stamp::Format() << std::endl; 
    std::cout << "stamp::Format(stamp::Now()) " << stamp::Format(stamp::Now()) << std::endl; 
    std::cout << "stamp::Format(0)            " << stamp::Format(0) << std::endl; 
    std::cout << "stamp::Format(1)            " << stamp::Format(1) << std::endl; 
    std::cout << "stamp::Format(1000)         " << stamp::Format(1000) << std::endl; 
    std::cout << "stamp::Format(1000000)      " << stamp::Format(1000000) << std::endl; 
    std::cout << "NOTE TIMEZONE LOCALIZATION OF THE ASSUMED UTC STAMPS" << std::endl ;  

    for(int i=0 ; i < 100 ; i++) 
    {
        uint64_t t = stamp::Now();
        std::cout << std::setw(5) << i << " : " << stamp::Format(t) << std::setw(9) << (t - t0) << std::endl ; 
    }
}

   
int main() 
{
    //test_reverse(); 
    test_format(); 
    return 0 ; 
}

