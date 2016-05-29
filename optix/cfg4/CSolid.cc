#include "CSolid.hh"
#include "CMath.hh"



// g4-
#include "G4VSolid.hh"
#include "G4VoxelLimits.hh"
#include "G4AffineTransform.hh"

// npy-
#include "NLog.hpp"

void CSolid::extent(const G4Transform3D& tran)
{
    G4AffineTransform  atran = CMath::make_affineTransform(tran);
    G4VoxelLimits      limit; // Unlimited

    G4double minX,maxX,minY,maxY,minZ,maxZ ;

    m_solid->CalculateExtent(kXAxis,limit,atran,minX,maxX);
    m_solid->CalculateExtent(kYAxis,limit,atran,minY,maxY);
    m_solid->CalculateExtent(kZAxis,limit,atran,minZ,maxZ);

    LOG(info) << "CSolid::extent"
              << " minX " << minX
              << " maxX " << maxX
              << " minY " << minY
              << " maxY " << maxY
              << " minZ " << minZ
              << " maxZ " << maxZ
              ;

}

