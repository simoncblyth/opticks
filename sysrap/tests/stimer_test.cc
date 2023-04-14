#include <iostream>
#include <iomanip>
#include <vector>
#include "stimer.h"

#include "NP.hh"

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

    using clock = std::chrono::system_clock ; 
    //using clock = std::chrono::steady_clock ; 
    //using clock = std::chrono::high_resolution_clock ; 

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

    //using clock = std::chrono::high_resolution_clock ; 
    using clock = std::chrono::system_clock ; 
    uint64_t ec = tim->start_count(); 
    std::chrono::time_point<clock> t0 = stimer::TimePoint(ec) ; 

    std::cout 
        << " ec " << ec 
        << " t0 " << stimer::Format(t0)
        << " ecf " << stimer::Format(ec) 
        << std::endl
        ;
 
}

void test_TimePoint_0()
{
    // hmm high_resolution_clock epochs dont travel between machines
    // try changing to system_clock epochs

    std::vector<uint64_t> ecs0 = { 1681415567555006257, 1681415570432562093, 1681415573040370404, 1681415575667773937, 1681415578398804926, 1681415581169340691, 1681415583816194221, 1681415586567097051, 1681415589753517102,
        1681415592426987792 } ; 

    std::vector<uint64_t> ecs1 = { 1681415825234793213, 1681415828042329959, 1681415831150709245, 1681415833581851217, 1681415836050784359, 1681415839417522516, 1681415842001793022, 1681415844497812015, 1681415846965829828,
        1681415850332486246 } ; 

    assert( ecs0.size() == ecs1.size() ); 

    for(int p=0 ; p < 2 ; p++)
    {
    for(int i=0 ; i < int(ecs0.size()) ; i++ )
    {
         uint64_t ec0 = ecs0[i] ; 
         uint64_t ec1 = ecs1[i] ; 

         if( p == 0 ) std::cout 
             << " ec0 " << std::setw(20) << ec0 
             << " ec1 " << std::setw(20) << ec1
             << std::endl 
             ;

         if( p == 1 ) std::cout 
             << " ec0 " << stimer::Format(ec0) 
             << " ec1 " << stimer::Format(ec1)
             << std::endl 
             ;
    }
    }

}


/**


Previously has to divide the Linux epoch count 
by 1000 to be comparable with the Darwin one.

Instead of such a hack, changed EpochCount to 
standardize on microsecond epoch counts. 
This makes the time stamps comparable modulo 7 hours time difference. 

N[blyth@localhost tests]$ ./stimer_test.sh 
 stimer::Format(1681469652582925) : Fri, 14.04.2023 18:54:12
 stimer::Format(1681467459953485) : Fri, 14.04.2023 18:17:39
 stimer::Format(1681467438445254) : Fri, 14.04.2023 18:17:18
N[blyth@localhost tests]$ date
Fri Apr 14 18:54:19 CST 2023
N[blyth@localhost tests]$ 

epsilon:tests blyth$ ./stimer_test.sh 
 stimer::Format(1681469695498380) : Fri, 14.04.2023 11:54:55
 stimer::Format(1681467459953485) : Fri, 14.04.2023 11:17:39
 stimer::Format(1681467438445254) : Fri, 14.04.2023 11:17:18
epsilon:tests blyth$ date
Fri Apr 14 11:55:00 BST 2023
epsilon:tests blyth$ 

**/


void test_EpochCountNow()
{
    std::vector<uint64_t> ecns = { stimer::EpochCountNow(), 1681467459953485, 1681467438445254316/1000 } ; 

    for(int i=0 ; i < int(ecns.size()) ; i++) 
    {
        uint64_t ecn = ecns[i] ; 
        std::cout 
            << " stimer::Format(" << ecn << ") : " << stimer::Format(ecn) 
            << std::endl 
            ; 
    }

}

void test_count()
{
    stimer* tim = stimer::create() ;

    stimer::sleep(1); 

    double dur = tim->done(); 

    std::cout << tim->desc() << std::endl ; 

    uint64_t t0 = tim->start_count(); 
    uint64_t t1 = tim->stop_count(); 
    std::cout 
        << " stimer::Format(t0) " << stimer::Format(t0) << " t0 " << t0 << std::endl 
        << " stimer::Format(t1) " << stimer::Format(t1) << " t1 " << t1 << std::endl  
        ; 


    NP* t = NP::Make<uint64_t>(2) ; 
    uint64_t* tt = t->values<uint64_t>() ; 
    tt[0] = t0 ; 
    tt[1] = t1 ; 
    t->save("$TTPATH"); 
}

/**
From BST (UTC+1) timezone the numpy presented times are corrected to UTC::

    epsilon:tests blyth$ ./stimer_test.sh 
    stimer::desc status STOPPED _start 1681480063800788 start Fri, 14.04.2023 14:47:43 _stop 1681480064801308 stop Fri, 14.04.2023 14:47:44 duration 1.000520e+00
     stimer::Format(t0) Fri, 14.04.2023 14:47:43 t0 1681480063800788
     stimer::Format(t1) Fri, 14.04.2023 14:47:44 t1 1681480064801308
    [1681480063800788 1681480064801308]

    np.c_[tt.view('datetime64[us]')]

    [['2023-04-14T13:47:43.800788']
     ['2023-04-14T13:47:44.801308']]

From CST (UTC+8) timezone the numpy presented times also corrected to UTC::

    N[blyth@localhost tests]$ ./stimer_test.sh 
    stimer::desc status STOPPED _start 1681480305503105 start Fri, 14.04.2023 21:51:45 _stop 1681480306505512 stop Fri, 14.04.2023 21:51:46 duration 1.002406e+00
     stimer::Format(t0) Fri, 14.04.2023 21:51:45 t0 1681480305503105
     stimer::Format(t1) Fri, 14.04.2023 21:51:46 t1 1681480306505512
    Python 3.7.7 (default, May  7 2020, 21:25:33) 
    Type 'copyright', 'credits' or 'license' for more information
    IPython 7.18.1 -- An enhanced Interactive Python. Type '?' for help.
    [1681480305503105 1681480306505512]

    np.c_[tt.view('datetime64[us]')]

    [['2023-04-14T13:51:45.503105']
     ['2023-04-14T13:51:46.505512']]

**/



int main()
{
    /*
    test_start_stop(); 
    test_done(); 
    test_lap(); 
    test_egg(); 
    test_desc(); 
    test_convert_0(); 
    test_convert_1(); 
    test_convert_2(); 
    test_convert_3(); 
    test_TimePoint_0();
    test_EpochCountNow(); 
    */

    test_count(); 

    return 0 ; 
}
