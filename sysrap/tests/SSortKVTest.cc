

#include "SSortKV.hh"

#include "SYSRAP_LOG.hh"
#include "PLOG.hh"


int main(int argc , char** argv )
{
    PLOG_(argc, argv);

    SYSRAP_LOG__ ; 


    bool descending = false ; 
    SSortKV skv(descending) ; 

    skv.add("red", 5.0);
    skv.add("green", 1.0);
    skv.add("blue", 3.0);

    skv.sort();
    skv.dump();


    return 0 ; 
}



