#include "GMaterial.hh"

GMaterial::GMaterial(const char* name, unsigned int index) : GPropertyMap(name, index, "material")
{
}

GMaterial::~GMaterial()
{
}

void GMaterial::Summary(const char* msg )
{
    GPropertyMap::Summary(msg);
}

