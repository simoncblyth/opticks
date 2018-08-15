
// g4-
#include "G4VSolid.hh"
#include "G4VoxelLimits.hh"
#include "G4AffineTransform.hh"

// npy-
#include "NGLM.hpp"
#include "NBoundingBox.hpp"
#include "NBBox.hpp"
#include "GLMFormat.hpp"

#include "X4SolidExtent.hh"
#include "X4AffineTransform.hh"

#include "PLOG.hh"



X4SolidExtent::X4SolidExtent(const G4VSolid* solid) 
   :
      m_solid(solid)
{
}

void X4SolidExtent::extent(const G4Transform3D& tran, glm::vec3& low, glm::vec3& high, glm::vec4& ce)
{
    G4AffineTransform  atran = X4AffineTransform::FromTransform(tran);
    G4VoxelLimits      limit; // Unlimited

    G4double minX,maxX,minY,maxY,minZ,maxZ ;

    m_solid->CalculateExtent(kXAxis,limit,atran,minX,maxX);
    m_solid->CalculateExtent(kYAxis,limit,atran,minY,maxY);
    m_solid->CalculateExtent(kZAxis,limit,atran,minZ,maxZ);

    low.x = float(minX) ;
    low.y = float(minY) ;
    low.z = float(minZ) ;

    high.x = float(maxX) ;
    high.y = float(maxY) ;
    high.z = float(maxZ) ;

    ce.x = float((minX + maxX)/2.) ; 
    ce.y = float((minY + maxY)/2.) ; 
    ce.z = float((minZ + maxZ)/2.) ; 
    ce.w = NBoundingBox::extent(low, high);


    LOG(debug) << "X4SolidExtent::extent"
              << " low " << gformat(low)
              << " high " << gformat(high)
              ;

}

nbbox* X4SolidExtent::Extent(const G4VSolid* solid)
{
    G4Transform3D tran ; 
    X4SolidExtent cs(solid); 
    nbbox bb ; 
    glm::vec4 ce ; 
    cs.extent( tran, bb.min, bb.max, ce ); 
    return new nbbox(bb) ;
}




