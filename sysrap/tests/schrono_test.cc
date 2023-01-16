// ./schrono_test.sh 

#include <iostream>
#include "schrono.h"

int main(int argc, char** argv)
{
    schrono::TP t0 = schrono::stamp(); 
    schrono::sleep(1); 
    schrono::TP t1 = schrono::stamp(); 
    double dt = schrono::duration(t0, t1 ); 
    std::cout << " dt " << std::scientific << dt << std::endl ; 

    std::time_t tt0 = schrono::approx_time_t(t0); 
    std::time_t tt1 = schrono::approx_time_t(t1); 

    std::cout 
        << " tt0 " << tt0 << " " << schrono::format(tt0) << std::endl 
        << " tt1 " << tt1 << " " << schrono::format(tt1) << std::endl 
        ; 

    return 0 ; 
}
