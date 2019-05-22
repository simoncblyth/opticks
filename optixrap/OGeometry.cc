#include "OGeometry.hh"

bool OGeometry::isGeometry() const
{
    return g.get() != NULL ; 
}
bool OGeometry::isGeometryTriangles() const
{
#if OPTIX_VERSION_MAJOR >= 6 
    return gt.get() != NULL ; 
#else
    return false ; 
#endif

}

