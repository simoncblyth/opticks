#include "OGeometry.hh"

bool OGeometry::isGeometry() const
{
    return g.get() != NULL ; 
}
bool OGeometry::isGeometryTriangles() const
{
    return gt.get() != NULL ; 
}

