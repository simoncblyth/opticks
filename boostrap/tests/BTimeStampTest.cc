// TEST=BTimeStampTest om-t

#include "BTimeStamp.hh"

#include <unistd.h>
#include <iostream>
#include <chrono>


#include "OPTICKS_LOG.hh"



void test_timing_1s()
{
    double t0 = BTimeStamp::RealTime() ; 
    auto s0 = std::chrono::high_resolution_clock::now();    

    sleep(1);
 
    double t1 = BTimeStamp::RealTime() ; 
    auto s1 = std::chrono::high_resolution_clock::now();    

    float dt = t1 - t0 ; 
    std::chrono::duration<double> s10 = s1 - s0;
    double ds = s10.count();     


    std::cerr << " BTimeStamp::RealTime() t0 " << t0 << std::endl ; 
    std::cerr << " BTimeStamp::RealTime() t1 " << t1 << std::endl ; 
    std::cerr << " dt = t1 - t0              " << dt << std::endl ; 
    std::cerr << " ds                        " << ds << std::endl ; 
}


void test_counting()
{
    int N = 1000000000 ; 

    while( N > 1 )
    {
        double t0 = BTimeStamp::RealTime() ; 

        int count = 0 ; for( int i=0 ; i < N ; i++ ) count+= 1 ; 

        double t1 = BTimeStamp::RealTime() ; 
        double dt = t1 - t0 ; 

        LOG(info) 
            << " N " << std::setw(10) << N   
            << " dt " <<  std::fixed << std::setprecision(9) << dt
            ; 

        N /= 10 ; 
    }
}


void test_counting_2()
{
    int M = 100000 ; 
    int N = M ; 

    double dt0 = -1 ; 
    double ds0 = -1 ; 

    while( N < 10*M )
    {
        auto s0 = std::chrono::high_resolution_clock::now();    
        double t0 = BTimeStamp::RealTime() ; 

        int count = 0 ; for( int i=0 ; i < N ; i++ ) count+= 1 ; 

        double t1 = BTimeStamp::RealTime() ; 
        auto s1 = std::chrono::high_resolution_clock::now();    
        double dt = t1 - t0 ; 
        std::chrono::duration<double> s10 = s1 - s0;
        double ds = s10.count();     
        if(dt0 < 0.) dt0 = dt ;   
        if(ds0 < 0.) ds0 = ds ;   


        LOG(info) 
            << " N " << std::setw(10) << N   
            << " dt " <<  std::fixed << std::setprecision(9) << dt
            << " dt/dt0 " <<  std::fixed << std::setprecision(9) << dt/dt0
            << " ds " <<  std::fixed << std::setprecision(9) << ds
            << " ds/ds0 " <<  std::fixed << std::setprecision(9) << ds/ds0
            ; 

        N += M ; 
    }
}






int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 
    LOG(info); 

    //test_timing_1s(); 
    //test_counting(); 
    test_counting_2(); 

    return 0 ; 
}
