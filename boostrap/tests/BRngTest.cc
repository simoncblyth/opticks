#include <iostream>
#include "BRng.hh"
#include "BRAP_LOG.hh"
#include "PLOG.hh"


void test_separate()
{

    BRng a(0,1, 42, "A") ; 
    a.dump();

    BRng b(0,1, 42, "B") ; 
    b.dump();


/*
    BRng c(0,100, 42, "C") ;
    c.dump();

    BRng d(0,100, 42, "D") ;
    d.dump();

    BRng e(0,1, 42, "E") ;
    e.dump();

    BRng f(0.9,1, 42, "F") ;
    f.dump();
*/
}



int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    BRAP_LOG__ ; 

    test_separate();


    return 0 ; 
}


