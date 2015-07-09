#include "Timer.hpp"
#include "Times.hpp"
#include "limits.h"

int main()
{

    Timer t ; 
    t.start();

    for(int i=0 ; i < INT_MAX/100 ; i++) if(i%1000000 == 0 ) printf(".");
    printf("\n");
 
    t("after loop");

    t.stop();
    t.dump();


    Times* ts = t.getTimes();
    ts->save("/tmp", "TimerTest.ini");




    return 0 ; 
}
