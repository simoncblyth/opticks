#include <iostream>


/*
#include <xercesc/util/PlatformUtils.hpp>
#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/sax/HandlerBase.hpp>
#include <xercesc/util/XMLUni.hpp>
#include <xercesc/dom/DOM.hpp>
*/

#include "OPTICKS_LOG.hh"
#include "SPath.hh"
#include "GDXML.hh"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    const char* srcpath = argc > 1 ? argv[1] : nullptr ; 
    const char* dstpath = SPath::Resolve("$TMP/GDXMLTest/test.gdml", FILEPATH) ; 

    if(!srcpath) 
    {
        std::cout << "Expecting a single argument /path/to/name.gdml " << std::endl ; 
        std::cout << "when issues are detected with the GDML a kludge fixed version is written to /path/to/name_GDXML.gdml " << std::endl ;   
        return 0 ; 
    }

    //xercesc::XMLPlatformUtils::Initialize();

    GDXML::Fix(dstpath, srcpath) ; 

    LOG(info)
        << " srcpath " << srcpath  
        << " dstpath " << dstpath  
        ; 

    return 0 ; 
}

