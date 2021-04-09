#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/util/PlatformUtils.hpp>
#include <xercesc/sax/HandlerBase.hpp>
#include <xercesc/util/XMLUni.hpp>
#include <xercesc/dom/DOM.hpp>

#include "GDMLKludgeFix.hh"

int main(int argc, char** argv)
{
    const char* srcpath = argc > 1 ? argv[1] : nullptr ; 

    if(!srcpath) return 0 ; 

    xercesc::XMLPlatformUtils::Initialize();

    GDMLKludgeFix kludgefix(srcpath) ; 

    return 0 ; 
}


