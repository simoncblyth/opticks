#include "CFG4_BODY.hh"

// g4-
#include "G4VSolid.hh"
#include "G4VoxelLimits.hh"
#include "G4AffineTransform.hh"

// npy-
#include "NGLM.hpp"
#include "NBoundingBox.hpp"
#include "GLMFormat.hpp"

#include "CMath.hh"
#include "CSolid.hh"

#include "PLOG.hh"


CSolid::CSolid(const G4VSolid* solid) 
   :
      m_solid(solid)
{
}


void CSolid::extent(const G4Transform3D& tran, glm::vec3& low, glm::vec3& high, glm::vec4& ce)
{
    G4AffineTransform  atran = CMath::make_affineTransform(tran);
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


    LOG(debug) << "CSolid::extent"
              << " low " << gformat(low)
              << " high " << gformat(high)
              ;

}





