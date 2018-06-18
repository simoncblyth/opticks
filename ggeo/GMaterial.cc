#include "GMaterial.hh"
#include "GDomain.hh"
#include "GPropertyLib.hh"

GMaterial::GMaterial(GMaterial* other, GDomain<float>* domain ) 
    : 
    GPropertyMap<float>(other, domain)
{
}

GMaterial::GMaterial(const char* name, unsigned int index) 
    : 
    GPropertyMap<float>(name, index, "material")
{
    init();
}

void GMaterial::init()
{
    setStandardDomain( GDomain<float>::GetDefaultDomain()) ;   
}

GMaterial::~GMaterial()
{
}

void GMaterial::Summary(const char* msg )
{
    GPropertyMap<float>::Summary(msg);
}



