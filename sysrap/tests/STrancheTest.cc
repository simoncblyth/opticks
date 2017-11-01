#include "STranche.hh"

#include "SYSRAP_LOG.hh"
#include "PLOG.hh"


int main(int argc , char** argv )
{
    PLOG_(argc, argv);

    SYSRAP_LOG__ ; 

   
    STranche st0( 1000, 100 );
    st0.dump();
 
    STranche st1( 1013, 100 );
    st1.dump();

    STranche st2( 1099, 100 );
    st2.dump();

    STranche st3( 1100, 100 );
    st3.dump();

    STranche st4( 1101, 100 );
    st4.dump();



    return 0  ; 
}
