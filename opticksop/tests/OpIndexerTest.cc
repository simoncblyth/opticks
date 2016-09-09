#include "OpIndexerApp.hh"

#include "SYSRAP_LOG.hh"  // headers for each projects logger
#include "BRAP_LOG.hh"
#include "NPY_LOG.hh"
#include "OKCORE_LOG.hh"
#include "OKOP_LOG.hh"

#include "PLOG.hh"

int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    SYSRAP_LOG__ ;     // setup loggers for all projects you want to see output from
    BRAP_LOG__ ;
    NPY_LOG__ ;
    OKCORE_LOG__ ;
    OKOP_LOG__ ;
    

    OpIndexerApp app(argc, argv) ;

    app.loadEvtFromFile();

    app.makeIndex();

    return 0 ; 
}
