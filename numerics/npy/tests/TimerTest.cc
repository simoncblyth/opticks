#include "Timer.hpp"
#include "limits.h"

int main()
{

    Timer t ; 
    t.start();

    for(int i=0 ; i < INT_MAX ; i++) if(i%1000000 == 0 ) printf(".");
    printf("\n");
 
    t("after loop");

    t.stop();
    t.dump();

    return 0 ; 
}
