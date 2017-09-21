#include "Opticks.hh"
#include "OpticksAna.hh"

#include "SYSRAP_LOG.hh"
#include "OKCORE_LOG.hh"
#include "PLOG.hh"

int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    SYSRAP_LOG__ ; 
    OKCORE_LOG__ ; 

    Opticks ok(argc, argv) ; 
    ok.configure();
    ok.ana();
  
    return ok.getRC();
}

/**

::

   OpticksAnaTest --anakey tpmt --tag 10 --cat PmtInBox


**/

