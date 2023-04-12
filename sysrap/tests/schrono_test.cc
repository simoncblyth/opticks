// ./schrono_test.sh 

#include <iostream>
#include "schrono.h"

void test_stamp_duration_approx_time()
{
    schrono::TP t0 = schrono::stamp(); 
    schrono::sleep(1); 
    schrono::TP t1 = schrono::stamp(); 
    double dt = schrono::duration(t0, t1 ); 
    std::cout << " dt " << std::scientific << dt << std::endl ; 

    std::time_t tt0 = schrono::approx_time_t(t0); 
    std::time_t tt1 = schrono::approx_time_t(t1); 

    std::cout 
        << " tt0 " << tt0 << " " << schrono::format(tt0) << " " << schrono::format(t0) << std::endl 
        << " tt1 " << tt1 << " " << schrono::format(tt1) << " " << schrono::format(t1) << std::endl 
        ; 
}

void test_delta_stamp()
{
    double dt0 = schrono::delta_stamp(); 
    schrono::sleep(1); 
    double dt1 = schrono::delta_stamp(); 

    std::cout << " dt0 " << std::scientific << dt0 << std::endl ; 
    std::cout << " dt1 " << std::scientific << dt1 << std::endl ; 

    std::chrono::duration<double> d(dt1) ; 
    //std::chrono::time_point<std::chrono::high_resolution_clock> td(d);


}





int main(int argc, char** argv)
{
    test_stamp_duration_approx_time() ; 
    /*
    test_delta_stamp(); 
    */


    return 0 ; 
}
