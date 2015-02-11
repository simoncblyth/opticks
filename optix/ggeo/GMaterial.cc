#include "GMaterial.hh"

GMaterial::GMaterial(const char* name) : GPropertyMap(name, "material")
{
}

GMaterial::~GMaterial()
{
}

void GMaterial::Summary(const char* msg )
{
    GPropertyMap::Summary(msg);
}

