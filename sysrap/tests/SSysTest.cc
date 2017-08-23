#include "SSys.hh"

#include "SYSRAP_LOG.hh"
#include "PLOG.hh"


int main(int argc , char** argv )
{
    PLOG_(argc, argv);

    SYSRAP_LOG__ ; 

    int rc = SSys::run("tpmt.py");
  
    LOG(info) << argv[0] << " rc " << rc ; 

    return 0  ; 
}
