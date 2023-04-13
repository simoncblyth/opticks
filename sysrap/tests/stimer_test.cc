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

struct Egg
{
   static stimer* TIMER ; 
   double boil(int seconds) ; 
};

stimer* Egg::TIMER = new stimer ; 

inline double Egg::boil(int seconds)
{
    TIMER->start()  ; 
    stimer::sleep(seconds); 
    return TIMER->done(); 
}

void test_egg()
{
    Egg d ; 

    double dt0 = d.boil(1) ; 
    std::cout << " dt0 " << dt0 << std::endl ; 

    double dt1 = d.boil(1) ; 
    std::cout << " dt1 " << dt1 << std::endl ; 

    double dt2 = d.boil(0) ; 
    std::cout << " dt2 " << dt2 << std::endl ; 
}

void test_desc()
{
    stimer* t = new stimer ; 
    std::cout << t->desc() << std::endl ; 

    t->start(); 
    std::cout << t->desc() << std::endl ; 

    stimer::sleep(1); 

    t->stop(); 
    std::cout << t->desc() << std::endl ; 

}


int main()
{
    /**
    test_start_stop(); 
    test_done(); 
    test_lap(); 
    test_egg(); 
    **/
    test_desc(); 


    return 0 ; 
}
