#include <climits>

#include "BTimeKeeper.hh"

#include "BTimes.hh"
#include "BTimesTable.hh"


int main()
{
    BTimeKeeper tk ; 
    tk.start();

    for(int i=0 ; i < INT_MAX/100 ; i++) if(i%1000000 == 0 ) printf(".");
    printf("\n");
 
    tk("after loop");

    tk.stop();


    BTimesTable* tt = tk.makeTable();
    tt->save("$TMP");

    tt->dump();


    return 0 ; 
}
