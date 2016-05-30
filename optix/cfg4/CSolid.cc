#include "CSolid.hh"
#include "CMath.hh"


// g4-
#include "G4VSolid.hh"
#include "G4VoxelLimits.hh"
#include "G4AffineTransform.hh"

// npy-
#include "NLog.hpp"
#include "NBoundingBox.hpp"
#include "GLMFormat.hpp"

void CSolid::extent(const G4Transform3D& tran, glm::vec3& low, glm::vec3& high, glm::vec4& ce)
{
    G4AffineTransform  atran = CMath::make_affineTransform(tran);
    G4VoxelLimits      limit; // Unlimited

    G4double minX,maxX,minY,maxY,minZ,maxZ ;

    m_solid->CalculateExtent(kXAxis,limit,atran,minX,maxX);
    m_solid->CalculateExtent(kYAxis,limit,atran,minY,maxY);
    m_solid->CalculateExtent(kZAxis,limit,atran,minZ,maxZ);

    low.x = minX ;
    low.y = minY ;
    low.z = minZ ;

    high.x = maxX ;
    high.y = maxY ;
    high.z = maxZ ;

    ce.x = (minX + maxX)/2.f ; 
    ce.y = (minY + maxY)/2.f ; 
    ce.z = (minZ + maxZ)/2.f ; 
    ce.w = NBoundingBox::extent(low, high);


    LOG(debug) << "CSolid::extent"
              << " low " << gformat(low)
              << " high " << gformat(high)
              ;

}





