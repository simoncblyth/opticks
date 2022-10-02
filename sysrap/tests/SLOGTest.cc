/**
SLOGTest.cc
=============


* I recall finding that the main level must larger than the libs
  for all the expected libs output to appear 
  ie that both level criteria gets applied 

  * TODO: confirm this, and see if some other organization might avoid that

* https://github.com/SergiusTheBest/plog/issues/72

::

    Also you can rewrite your previous sample:

    plog::init<1000>(plog::verbose, "/var/log/my.log");
    plog::init<0>(plog::info, plog::get<1000>());
    plog::init<1>(plog::debug, plog::get<1000>());
    plog::init<2>(plog::warning, plog::get<1000>());

    so the default log stays 0 and the sink becomes 1000.



**/


#include "OPTICKS_LOG.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    std::cout << "SLOG::Flags() " << SLOG::Flags() << std::endl ; 


    plog::Logger<0>* main_logger = plog::get<0>() ; 

    plog::Severity main_level = main_logger->getMaxSeverity(); 
    std::cout 
        << " main_logger  " << main_logger 
        << " plog::severityToString(main_level) " << plog::severityToString(main_level) 
        << std::endl  
        ; 

    std::cout 
        << " SLOG::Desc<0>(main_logger) " << SLOG::Desc<0>(main_logger) 
        << std::endl  
        << " SLOG::Desc<0>() lib_logger " << SLOG::Desc<0>() 
        << std::endl  
        ; 


    bool change_main_level = false ; 
    if( change_main_level  )
    {
        main_logger->setMaxSeverity(info) ; 
        std::cout 
            << " SLOG::Desc<0>(main_logger) " << SLOG::Desc<0>(main_logger) 
            << " : after main_logger->setMaxSeverity(info) "
            << std::endl  
            ; 
    }


    std::cout << " [ logging from main " << std::endl ; 
    LOG(none)    << " LOG(none) " ; 
    LOG(fatal)   << " LOG(fatal) " ; 
    LOG(error)   << " LOG(error) " ; 
    LOG(warning) << " LOG(warning) " ; 
    LOG(info)    << " LOG(info) " ; 
    LOG(debug)   << " LOG(debug) " ; 
    LOG(verbose) << " LOG(verbose) " ; 
    std::cout << " ] logging from main " << std::endl ; 


    std::cout << " [ logging from lib " << std::endl ; 
    SLOG::Dump(); 
    std::cout << " ] logging from lib " << std::endl ; 

    return 0 ; 
}
