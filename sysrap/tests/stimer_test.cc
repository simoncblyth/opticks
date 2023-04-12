#include <iostream>
#include "stimer.h"

void test_start_stop()
{
    stimer* t = new stimer  ; 
    t->start(); 
    stimer::sleep(1) ; 
    t->stop(); 
    double dt0 = t->duration(); 
    std::cout << " dt0 " << dt0 << std::endl ; 

    t->start(); 
    t->stop(); 
    double dt1 = t->duration(); 

    std::cout << " dt1 " << dt1 << std::endl ; 
}

void test_done()
{
    stimer* t = stimer::create(); 
    stimer::sleep(1); 
    double dt0 = t->done(); 
    std::cout << " dt0 " << dt0 << std::endl ; 
    t->start(); 
    stimer::sleep(1); 
    double dt1 = t->done(); 
    std::cout << " dt1 " << dt1 << std::endl ; 
}


void test_lap()
{
    stimer* t = stimer::create(); 
    stimer::sleep(1); 
    double dt0 = t->lap(); 
    std::cout << " dt0 " << dt0 << std::endl ; 
    stimer::sleep(1); 
    double dt1 = t->lap(); 
    std::cout << " dt1 " << dt1 << std::endl ; 
}

int main()
{
    /**
    test_start_stop(); 
    test_done(); 
    **/
    test_lap(); 
    return 0 ; 
}
