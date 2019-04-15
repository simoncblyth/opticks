#include "X4GDMLWrite.hh"


#include <xercesc/dom/DOM.hpp>
#include <xercesc/util/XMLString.hpp>
#include <xercesc/util/PlatformUtils.hpp>
#include <xercesc/framework/LocalFileFormatTarget.hpp>


X4GDMLWrite::X4GDMLWrite()
{
    xercesc::XMLPlatformUtils::Initialize();
}


void X4GDMLWrite::write(const G4VSolid* solid)
{
}




