#include <climits>

#include "Timer.hpp"
#include "Times.hpp"
#include "TimesTable.hpp"


int main()
{

    Timer t ; 
    t.start();

    for(int i=0 ; i < INT_MAX/100 ; i++) if(i%1000000 == 0 ) printf(".");
    printf("\n");
 
    t("after loop");

    t.stop();


    TimesTable* tt = t.makeTable();
    tt->save("/tmp");

    tt->dump();


    return 0 ; 
}
