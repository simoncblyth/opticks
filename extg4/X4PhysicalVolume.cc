#include <iostream>
#include <sstream>
#include <iomanip>

#include "G4VPhysicalVolume.hh"
#include "G4LogicalVolume.hh"
#include "G4Material.hh"
#include "G4VSolid.hh"
#include "G4TransportationManager.hh"

#include "X4PhysicalVolume.hh"
#include "X4Material.hh"
#include "X4Transform3D.hh"

#include "GGeo.hh"
#include "GMaterial.hh"
#include "GMaterialLib.hh"
#include "BOpticksKey.hh"
#include "Opticks.hh"
#include "SDigest.hh"
#include "PLOG.hh"


const G4VPhysicalVolume* const X4PhysicalVolume::Top()
{
    const G4VPhysicalVolume* const top = G4TransportationManager::GetTransportationManager()->GetNavigatorForTracking()->GetWorldVolume() ;
    return top ; 
}

GGeo* X4PhysicalVolume::Convert(const G4VPhysicalVolume* const top)
{
    X4PhysicalVolume pv(top) ;  
    GGeo* gg = pv.getGGeo();
    return gg ; 
}

X4PhysicalVolume::X4PhysicalVolume(const G4VPhysicalVolume* const top)
    :
    m_top(top),
    m_id(Id(m_top)),
    m_setid(BOpticksKey::SetKey(m_id)),
    m_ok(Opticks::GetOpticks()),  // hmm need to set the idpath appropriately based on identity of the volume ?
    m_ggeo(new GGeo(m_ok)),
    m_mlib(m_ggeo->getMaterialLib()),
    m_verbosity(m_ok->getVerbosity()),
    m_pvcount(0)
{
    init();
}

GGeo* X4PhysicalVolume::getGGeo()
{
    return m_ggeo ; 
}

void X4PhysicalVolume::init()
{
    VolumeTreeTraverse();
}

void X4PhysicalVolume::VolumeTreeTraverse()
{
     assert(m_top) ;
     G4LogicalVolume* lv = m_top->GetLogicalVolume() ;
     VolumeTreeTraverse(lv, 0 );
}

std::string X4PhysicalVolume::Digest( const G4LogicalVolume* const lv, const G4int depth )
{
    SDigest dig ;

    for (unsigned i=0; i < unsigned(lv->GetNoDaughters()) ; i++)
    {
        const G4VPhysicalVolume* const d_pv = lv->GetDaughter(i);

        G4RotationMatrix rot, invrot;

        if (d_pv->GetFrameRotation() != 0)
        {
           rot = *(d_pv->GetFrameRotation());
           invrot = rot.inverse();
        }

        std::string d_dig = Digest(d_pv->GetLogicalVolume(),depth+1);

        // postorder visit region is here after the recursive call

        G4Transform3D P(invrot,d_pv->GetObjectTranslation());

        std::string p_dig = X4Transform3D::Digest(P) ; 
    
        dig.update( const_cast<char*>(d_dig.data()), d_dig.size() );  
        dig.update( const_cast<char*>(p_dig.data()), p_dig.size() );  
    }

    // Avoid pointless repetition of full material digests for every 
    // volume by digesting just the material name (could use index instead)
    // within the recursion.
    //
    // Full material digests of all properties are included after the recursion.

    G4Material* material = lv->GetMaterial();
    const G4String& name = material->GetName();    
    dig.update( const_cast<char*>(name.data()), name.size() );  

    return dig.finalize();
}


std::string X4PhysicalVolume::Digest( const G4VPhysicalVolume* const top)
{
    const G4LogicalVolume* lv = top->GetLogicalVolume() ;
    std::string tree = Digest(lv, 0 ); 
    std::string mats = X4Material::Digest(); 

    SDigest dig ;
    dig.update( const_cast<char*>(tree.data()), tree.size() );  
    dig.update( const_cast<char*>(mats.data()), mats.size() );  
    return dig.finalize();
}


const char* X4PhysicalVolume::Id(const G4VPhysicalVolume* const top )
{
    std::string digest = Digest(top);

    const char* exename = PLOG::instance->args.exename() ; 

    std::stringstream ss ; 
    ss 
       << exename
       << "."
       << "X4PhysicalVolume"
       << "."
       << top->GetName()
       << "."
       << digest 
       ;
       
    std::string id = ss.str();
    return strdup(id.c_str());
}   



G4Transform3D X4PhysicalVolume::VolumeTreeTraverse(const G4LogicalVolume* const lv, const G4int depth)
{
     G4Transform3D R, invR ;  // huh invR stays identity, see g4dae/src/G4DAEWriteStructure.cc
     Visit(lv);

     const G4int daughterCount = lv->GetNoDaughters();
     for (G4int i=0;i<daughterCount;i++)
     {
         const G4VPhysicalVolume* const physvol = lv->GetDaughter(i);

         G4Transform3D daughterR;

         G4RotationMatrix rot, invrot;
         if (physvol->GetFrameRotation() != 0)
         {
            rot = *(physvol->GetFrameRotation());
            invrot = rot.inverse();
         }

         daughterR = VolumeTreeTraverse(physvol->GetLogicalVolume(),depth+1);

         // G4Transform3D P(rot,physvol->GetObjectTranslation());  GDML does this : not inverting the rotation portion 
         G4Transform3D P(invrot,physvol->GetObjectTranslation());

         VisitPV(physvol, invR*P*daughterR);  // postorder (visit after recursive call)

        // This mimicks what is done in g4dae/src/G4DAEWriteStructure.cc which follows GDML (almost) 
        // despite trying to look like it accounts for all the transforms through the tree
        // it aint doing that as:
        //
        //   * R and invR always stay identity -> daughterR is always identity 
        //   * so only P is relevant, which is in inverse of the frame rotation and the object translation   
        //
        //  So the G4DAE holds just the one level transforms, relying on the post processing
        //  tree traverse to multiply them to give global transforms 
     }


     G4Material* material = lv->GetMaterial();
     G4MaterialPropertiesTable* mpt = material->GetMaterialPropertiesTable();
     assert( mpt );
     const std::string& matname_ = material->GetName(); 
     const char* matname = matname_.c_str() ; 

     if(!m_mlib->hasMaterial(matname))
     { 
         GMaterial* mat = X4Material::Convert(material) ; 
         unsigned index = m_mlib->getNumMaterials();
         mat->setIndex( index ); 
         m_ggeo->add(mat);
     }
     return R ; 
}



void X4PhysicalVolume::Visit(const G4LogicalVolume* const lv)
{
    const G4String lvname = lv->GetName();
    G4VSolid* solid = lv->GetSolid();
    G4Material* material = lv->GetMaterial();

    const G4String geoname = solid->GetName() ;
    const G4String matname = material->GetName();

    if(m_verbosity > 1 )
        LOG(info) << "X4PhysicalVolume::Visit"
               << std::setw(20) << lvname
               << std::setw(50) << geoname
               << std::setw(20) << matname
               ;
}


void X4PhysicalVolume::VisitPV(const G4VPhysicalVolume* const pv, const G4Transform3D& T )
{
    LOG(debug) << "X4PhysicalVolume::VisitPV"
              << " pvcount " << std::setw(6) << m_pvcount
              << " pvname " << pv->GetName()
              ;
    m_pvcount += 1 ;

}


