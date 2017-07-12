#include <iostream>
#include "BOpticksResource.hh"
#include "BRAP_LOG.hh"
#include "PLOG.hh"

int main(int argc, char** argv)
{
    PLOG_(argc, argv);

    BRAP_LOG__ ; 

    BOpticksResource res ; 

    res.Summary();


    std::string treedir = res.getDebuggingTreedir(argc, argv);  //  requires the debugging only IDPATH envvar

    std::cout << argv[0]
              << " treedir " << treedir
              << std::endl 
              ;


    return 0 ; 
}
