#include <iostream>
#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/util/PlatformUtils.hpp>
#include <xercesc/sax/HandlerBase.hpp>
#include <xercesc/util/XMLUni.hpp>
#include <xercesc/dom/DOM.hpp>


#include "OPTICKS_LOG.hh"
#include "CGDMLKludge.hh"


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    const char* srcpath = argc > 1 ? argv[1] : nullptr ; 

    if(!srcpath) 
    {
        std::cout << "Expecting a single argument /path/to/name.gdml " << std::endl ; 
        std::cout << "when issues are detected with the GDML a kludge fixed version is written to /path/to/name_CGDMLKludge.gdml " << std::endl ;   
        return 0 ; 
    }

    xercesc::XMLPlatformUtils::Initialize();

    const char* dstpath = CGDMLKludge::Fix(srcpath) ; 
    LOG(info)
        << "CGDMLKludge::Fix(" << srcpath << ") " 
        << ( dstpath ? "FIXED ISSUES AND WROTE A KLUDGED GDML FILE:" : "found no issues to fix" )
        << ( dstpath ? dstpath : " " )
        ; 

    return 0 ; 
}


