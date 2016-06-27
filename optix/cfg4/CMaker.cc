#include "CMaker.hh"

// npy-
#include "NGLM.hpp"

// ggeo-
#include "GCSG.hh"

// g4-
#include "G4Sphere.hh"
#include "G4Box.hh"
#include "G4Tubs.hh"

#include "G4RotationMatrix.hh"
#include "G4Transform3D.hh"

#include "G4UnionSolid.hh"
#include "G4IntersectionSolid.hh"

#include "globals.hh"
#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"

#include "PLOG.hh"



CMaker::CMaker(Opticks* cache, int verbosity) 
   :
   m_cache(cache),
   m_verbosity(verbosity)
{
}   


std::string CMaker::LVName(const char* shapename)
{
    std::stringstream ss ; 
    ss << shapename << "_log" ; 
    return ss.str();
}

std::string CMaker::PVName(const char* shapename)
{
    std::stringstream ss ; 
    ss << shapename << "_phys" ; 
    return ss.str();
}


G4VSolid* CMaker::makeSphere(const glm::vec4& param)
{
    G4double radius = param.w*mm ; 
    G4Sphere* solid = new G4Sphere("sphere_solid", 0., radius, 0., twopi, 0., pi);  
    return solid ; 
}

G4VSolid* CMaker::makeBox(const glm::vec4& param)
{
    G4double extent = param.w*mm ; 
    G4double x = extent;
    G4double y = extent;
    G4double z = extent;
    G4Box* solid = new G4Box("box_solid", x,y,z);
    return solid ; 
}

G4VSolid* CMaker::makeSolid(char shapecode, const glm::vec4& param)
{
    G4VSolid* solid = NULL ; 
    switch(shapecode)
    {
        case 'B':solid = makeBox(param);break;
        case 'S':solid = makeSphere(param);break;
    }
    return solid ; 
} 


G4VSolid* CMaker::makeSolid(GCSG* csg, unsigned int index)
{
   // hmm this is somewhat specialized to known structure of DYB PMT

    unsigned int nc = csg->getNumChildren(index); 
    unsigned int fc = csg->getFirstChildIndex(index); 
    unsigned int lc = csg->getLastChildIndex(index); 
    unsigned int tc = csg->getTypeCode(index);
    const char* tn = csg->getTypeName(index);

    if(m_verbosity>0)
    LOG(info) 
           << "CMaker::makeSolid "
           << "  i " << std::setw(2) << index  
           << " nc " << std::setw(2) << nc 
           << " fc " << std::setw(2) << fc 
           << " lc " << std::setw(2) << lc 
           << " tc " << std::setw(2) << tc 
           << " tn " << tn 
           ;

   G4VSolid* solid = NULL ; 

   if(csg->isUnion(index))
   {
       assert(nc == 2);
       std::stringstream ss ; 
       ss << "union-ab" 
          << "-i-" << index
          << "-fc-" << fc 
          << "-lc-" << lc 
          ;
       std::string ab_name = ss.str();

       int a = fc ; 
       int b = lc ; 

       G4ThreeVector apos(csg->getX(a)*mm, csg->getY(a)*mm, csg->getZ(a)*mm); 
       G4ThreeVector bpos(csg->getX(b)*mm, csg->getY(b)*mm, csg->getZ(b)*mm);

       G4RotationMatrix ab_rot ; 
       G4Transform3D    ab_transform(ab_rot, bpos  );

       G4VSolid* asol = makeSolid(csg, a );
       G4VSolid* bsol = makeSolid(csg, b );

       G4UnionSolid* uso = new G4UnionSolid( ab_name.c_str(), asol, bsol, ab_transform );
       solid = uso ; 
   }
   else if(csg->isIntersection(index))
   {
       assert(nc == 3 && fc + 2 == lc );

       std::string ij_name ;      
       std::string ijk_name ;      

       {
          std::stringstream ss ; 
          ss << "intersection-ij" 
              << "-i-" << index 
              << "-fc-" << fc 
              << "-lc-" << lc 
              ;
          ij_name = ss.str();
       }
  
       {
          std::stringstream ss ; 
          ss << "intersection-ijk" 
              << "-i-" << index 
              << "-fc-" << fc 
              << "-lc-" << lc 
              ;
          ijk_name = ss.str();
       }


       int i = fc + 0 ; 
       int j = fc + 1 ; 
       int k = fc + 2 ; 

       G4ThreeVector ipos(csg->getX(i)*mm, csg->getY(i)*mm, csg->getZ(i)*mm); // kinda assumed 0,0,0
       G4ThreeVector jpos(csg->getX(j)*mm, csg->getY(j)*mm, csg->getZ(j)*mm);
       G4ThreeVector kpos(csg->getX(k)*mm, csg->getY(k)*mm, csg->getZ(k)*mm);

       G4VSolid* isol = makeSolid(csg, i );
       G4VSolid* jsol = makeSolid(csg, j );
       G4VSolid* ksol = makeSolid(csg, k );

       G4RotationMatrix ij_rot ; 
       G4Transform3D    ij_transform(ij_rot, jpos  );
       G4IntersectionSolid* ij_sol = new G4IntersectionSolid( ij_name.c_str(), isol, jsol, ij_transform  );

       G4RotationMatrix ijk_rot ; 
       G4Transform3D ijk_transform(ijk_rot,  kpos );
       G4IntersectionSolid* ijk_sol = new G4IntersectionSolid( ijk_name.c_str(), ij_sol, ksol, ijk_transform  );

       solid = ijk_sol ; 
   } 
   else if(csg->isSphere(index))
   {
        std::stringstream ss ; 
        ss << "sphere" 
              << "-i-" << index 
              ; 

       std::string sp_name = ss.str();

       float inner = float(csg->getInnerRadius(index)*mm) ;
       float outer = float(csg->getOuterRadius(index)*mm) ;
       float startTheta = float(csg->getStartTheta(index)*pi/180.) ;
       float deltaTheta = float(csg->getDeltaTheta(index)*pi/180.) ;

       assert(outer > 0 ) ; 

       float startPhi = 0.f ; 
       float deltaPhi = 2.f*float(pi) ; 

       LOG(info) << "CMaker::makeSolid csg Sphere"
                 << " inner " << inner 
                 << " outer " << outer
                 << " startTheta " << startTheta
                 << " deltaTheta " << deltaTheta
                 << " endTheta " << startTheta + deltaTheta
                 ;
 
       solid = new G4Sphere( sp_name.c_str(), inner > 0 ? inner : 0.f , outer, startPhi, deltaPhi, startTheta, deltaTheta  );

   }
   else if(csg->isTubs(index))
   {
        std::stringstream ss ; 
        ss << "tubs" 
              << "-i-" << index 
              ; 

       std::string tb_name = ss.str();
       float inner = 0.f ; // csg->getInnerRadius(i); kludge to avoid rejig as sizeZ occupies innerRadius spot
       float outer = float(csg->getOuterRadius(index)*mm) ;
       float sizeZ = float(csg->getSizeZ(index)*mm) ;   // half length   
       sizeZ /= 2.f ;   

       // PMT base looks too long without the halfing (as seen by photon interaction position), 
       // but tis contrary to manual http://lhcb-comp.web.cern.ch/lhcb-comp/Frameworks/DetDesc/Documents/Solids.pdf

       assert(sizeZ > 0 ) ; 

       float startPhi = 0.f ; 
       float deltaPhi = 2.f*float(pi) ; 

       if(m_verbosity>0)
       LOG(info) << "CMaker::makeSolid"
                 << " name " << tb_name
                 << " inner " << inner 
                 << " outer " << outer 
                 << " sizeZ " << sizeZ 
                 << " startPhi " << startPhi
                 << " deltaPhi " << deltaPhi
                 << " mm " << mm
                 ;

       solid = new G4Tubs( tb_name.c_str(), inner > 0 ? inner : 0.f , outer, sizeZ, startPhi, deltaPhi );

   }
   else
   {
       LOG(warning) << "CMaker::makeSolid implementation missing " ; 
   }

   assert(solid) ; 
   return solid ; 
}


