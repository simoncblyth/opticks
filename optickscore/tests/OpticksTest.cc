// ggv --opticks 

#include <iostream>

#include "PLOG.hh"
#include "PLOG_INIT.hh"

#include "BRAP_LOG.hh"
#include "NPY_LOG.hh"
#include "OKCORE_LOG.hh"

#include <plog/Appenders/ColorConsoleAppender.h>
//#include <plog/Appenders/ConsoleAppender.h>


#include "Opticks.hh"


int main(int argc, char** argv)
{
    static plog::ColorConsoleAppender<plog::TxtFormatter> consoleAppender; 
    PLOG_INIT( &consoleAppender, PLOG(argc,argv).level );
    LOG(info) << argv[0] ;


    plog::Logger<0>* logger = plog::get(); 

    BRAP_LOG::Initialize(logger, logger->getMaxSeverity() );
    BRAP_LOG::Check("main -> BRAP_LOG");

    NPY_LOG::Initialize(logger, logger->getMaxSeverity() );
    NPY_LOG::Check("main -> NPY_LOG");

    OKCORE_LOG::Initialize(logger, logger->getMaxSeverity() );
    OKCORE_LOG::Check("main -> OKCORE_LOG");





    Opticks ok(argc, argv);

    //ok.Summary();


    ok.configure();
    LOG(info) << "OpticksTest::main aft configure" ;

    unsigned long long seqmat = 0x0123456789abcdef ;

    std::string s_seqmat = Opticks::MaterialSequence(seqmat) ;

    LOG(info) 
              << "OpticksTest::main"
              << " seqmat "
              << std::hex << seqmat << std::dec
              << " MaterialSequence " 
              << s_seqmat
              ;

    return 0 ;
}
