#include "GMaterial.hh"


// TODO: move default domain down into OKCORE okc-
#include "GPropertyLib.hh"


GMaterial::GMaterial(GMaterial* other) : GPropertyMap<float>(other)
{
}

GMaterial::GMaterial(const char* name, unsigned int index) : GPropertyMap<float>(name, index, "material")
{
   init();
}

GMaterial::~GMaterial()
{
}

void GMaterial::Summary(const char* msg )
{
    GPropertyMap<float>::Summary(msg);
}


void GMaterial::init()
{
    GDomain<float>* sd = GPropertyLib::getDefaultDomain();
    setStandardDomain(sd);
}



