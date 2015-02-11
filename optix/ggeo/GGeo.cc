#include "GGeo.hh"
#include "GSolid.hh"

GGeo::GGeo()
{
}

GGeo::~GGeo()
{
}

void GGeo::addSolid(GSolid* solid)
{
    m_solids.push_back(solid);
}






