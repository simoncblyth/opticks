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


void test_convert_0()
{
    // https://stackoverflow.com/questions/31255486/how-do-i-convert-a-stdchronotime-point-to-long-and-back

    using namespace std::chrono;
    auto now = system_clock::now();
    auto now_ms = time_point_cast<milliseconds>(now);

    auto value = now_ms.time_since_epoch();
    long duration = value.count();

    milliseconds dur(duration);

    time_point<system_clock> dt(dur);

    if (dt != now_ms)
        std::cout << "Failure." << std::endl;
    else
        std::cout << "Success." << std::endl;
}


void test_convert_1()
{
    std::chrono::time_point<std::chrono::system_clock> t0 = std::chrono::system_clock::now();

    unsigned long long ns0 = t0.time_since_epoch().count();
    std::cout << " ns0 " << ns0 << std::endl ; 

    std::chrono::system_clock::duration d(ns0) ; 
    std::chrono::system_clock::time_point t1(d);

    unsigned long long ns1 = t1.time_since_epoch().count();
    std::cout << " ns1 " << ns1 << std::endl ; 

    std::chrono::duration<double> dt0 = t0.time_since_epoch() ; 
    std::chrono::duration<double> dt1 = t1.time_since_epoch() ; 
    double _dt0 =  dt0.count() ; 
    double _dt1 =  dt1.count() ; 
    
    std::cout << " _dt0 " << std::scientific << _dt0 << std::endl ; 
    std::cout << " _dt1 " << std::scientific << _dt1 << std::endl ; 
}

void test_convert_2()
{
    // working out how to convert a time_point into POD and back 

    //using clock = std::chrono::system_clock ; 
    //using clock = std::chrono::steady_clock ; 
    using clock = std::chrono::high_resolution_clock ; 

    std::chrono::time_point<clock> t0 = clock::now();
    std::cout << " t0 : " << stimer::Format(t0) << std::endl ; 

    uint64_t ns0 = t0.time_since_epoch().count();
    std::cout << " ns0 " << ns0 << std::endl ; 

    clock::duration d(ns0) ; 
    clock::time_point t1(d);

    std::cout << " t1 : " << stimer::Format(t1) << std::endl ; 


    uint64_t ns1 = t1.time_since_epoch().count();
    std::cout << " ns1 " << ns1 << std::endl ; 

    std::chrono::duration<double> dt0 = t0.time_since_epoch() ; 
    std::chrono::duration<double> dt1 = t1.time_since_epoch() ; 
    double _dt0 =  dt0.count() ; 
    double _dt1 =  dt1.count() ; 
    
    std::cout << " _dt0 " << std::scientific << _dt0 << std::endl ; 
    std::cout << " _dt1 " << std::scientific << _dt1 << std::endl ; 
}

void test_convert_3()
{
    stimer* tim = new stimer ; 
    tim->start(); 

    using clock = std::chrono::high_resolution_clock ; 
    uint64_t ec = tim->start_count(); 
    std::chrono::time_point<clock> t0 = stimer::TimePoint(ec) ; 

    std::cout 
        << " ec " << ec 
        << " t0 " << stimer::Format(t0)
        << " ecf " << stimer::Format(ec) 
        << std::endl
        ;
 
}


int main()
{
    /**
    test_start_stop(); 
    test_done(); 
    test_lap(); 
    test_egg(); 
    test_desc(); 
    test_convert_0(); 
    test_convert_1(); 
    test_convert_2(); 
    **/

    test_convert_3(); 

    return 0 ; 
}
