// name=schrono_test ; gcc $name.cc -std=c++11 -lstdc++ -I.. -o /tmp/$name && /tmp/$name

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
        << " tt0 " << tt0 << std::endl 
        << " tt1 " << tt1 << std::endl 
        ; 


    std::tm* tm = std::localtime(&tt0);
    char buffer[32];
    // Format: Mo, 15.06.2009 20:20:00
    std::strftime(buffer, 32, "%a, %d.%m.%Y %H:%M:%S", tm);  

    std::cout << buffer << std::endl ;   


    return 0 ; 
}
