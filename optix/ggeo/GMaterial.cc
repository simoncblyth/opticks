#include "GMaterial.hh"


GMaterial::GMaterial(GMaterial* other) : GPropertyMap<float>(other)
{
}

GMaterial::GMaterial(const char* name, unsigned int index) : GPropertyMap<float>(name, index, "material")
{
}

GMaterial::~GMaterial()
{
}

void GMaterial::Summary(const char* msg )
{
    GPropertyMap<float>::Summary(msg);
}





