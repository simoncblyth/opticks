#include <climits>

#include "Timer.hpp"

#include "BTimes.hh"
#include "BTimesTable.hh"


int main()
{

    Timer t ; 
    t.start();

    for(int i=0 ; i < INT_MAX/100 ; i++) if(i%1000000 == 0 ) printf(".");
    printf("\n");
 
    t("after loop");

    t.stop();


    BTimesTable* tt = t.makeTable();
    tt->save("$TMP");

    tt->dump();


    return 0 ; 
}
