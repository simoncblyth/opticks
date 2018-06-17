#include "X4.hh"
#include "G4Material.hh"
#include "G4LogicalSurface.hh"
#include "BStr.hh"

const char* X4::ShortName( const G4Material* const material )
{
    if(material == NULL) return NULL ; 
    const std::string& name = material->GetName();
    return ShortName(name);
}

const char* X4::ShortName( const G4LogicalSurface* const surface )
{
    if(surface == NULL) return NULL ; 
    const std::string& name = surface->GetName();
    return ShortName(name);
}

const char* X4::ShortName( const std::string& name )
{
    char* shortname = BStr::trimPointerSuffixPrefix(name.c_str(), NULL) ;  
    return strdup( shortname );
}


