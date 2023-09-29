// ./stamp_test.sh 

#include <chrono>
#include <iostream>
#include <iomanip>

#include "stamp.h"
#include "NP.hh"


/**
test_reverse
-------------

1. creates uint64_t timestamp t0 using stamp::Now
2. uses that integer to create std::chrono::time_point and gets the count 
3. asserts that they match 

Success of this test means can safely store time stamps at uint64_t values

**/

void test_reverse()
{
    std::cout << std::endl << "test_reverse " << std::endl ; 
    uint64_t t0 = stamp::Now(); 

    using Clock = std::chrono::system_clock;
    using Unit  = std::chrono::microseconds  ; 

    std::chrono::time_point<Clock> t0_{Unit{t0}} ; 
    uint64_t t1 = std::chrono::duration_cast<Unit>(t0_.time_since_epoch()).count() ; 

    std::cout << "t0: " << t0 <<  std::endl ; 
    std::cout << "t1: " << t1 <<  std::endl ; 

    assert( t0 == t1 ) ; 
}


/**
test_performance
----------------

1. creates an array able to hold 25 million uint64_t values 
2. fills the array with timestamps 
3. saves the array to filepath $FOLD/tt.npy 

On old laptop: counting and stamp recording to 25M takes roughly 1s (1M us)
so 25 stamp records takes ~1us  
(microsecond, 10-6 second, one millionth of a second)
 
**/

void test_performance()
{
    static const int N = 1000000*25 ;  
    const char* ttpath = "$FOLD/tt.npy" ; 

    std::cout << std::endl << "test_performance : N " << N  << " : save stamps to ttpath " << ttpath << std::endl ; 

    NP* t = NP::Make<uint64_t>(N) ;  
    uint64_t* tt = t->values<uint64_t>(); 
    for(int i=0 ; i < N ; i++) tt[i] = stamp::Now(); 

    t->set_meta<std::string>("IDENTITY", U::GetEnv("IDENTITY", "no-IDENTITY") );     
    t->save(ttpath);  
}

/**
test_format
-------------

1. create timestamp uint64_t 
2. exercise stamp::Format using that timestamp and other values

**/


void test_format()
{
    std::cout << std::endl << "test_format" << std::endl ; 
    uint64_t t0 = stamp::Now(); 
    std::cout << t0 << " : " << stamp::Format(t0) << " " << stamp::Format() << std::endl ; 

    std::cout << "stamp::Format()             " << stamp::Format() << std::endl; 
    std::cout << "stamp::Format(stamp::Now()) " << stamp::Format(stamp::Now()) << std::endl; 
    std::cout << "stamp::Format(0)            " << stamp::Format(0) << std::endl; 
    std::cout << "stamp::Format(1)            " << stamp::Format(1) << std::endl; 
    std::cout << "stamp::Format(1000)         " << stamp::Format(1000) << std::endl; 
    std::cout << "stamp::Format(1000000)      " << stamp::Format(1000000) << std::endl; 
    std::cout << "NOTE TIMEZONE LOCALIZATION OF THE ASSUMED UTC STAMPS" << std::endl ;  

    for(int i=0 ; i < 10 ; i++) 
    {
        uint64_t t = stamp::Now();
        std::cout << std::setw(5) << i << " : " << stamp::Format(t) << std::setw(9) << (t - t0) << std::endl ; 
    }
}

   
int main() 
{
    test_reverse(); 
    test_performance(); 
    test_format(); 
    return 0 ; 
}

