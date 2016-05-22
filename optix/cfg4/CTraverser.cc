#include "CTraverser.hh"

#include <algorithm>
#include <sstream>

#include "G4VPhysicalVolume.hh"
#include "G4LogicalVolume.hh"
#include "G4VSolid.hh"
#include "G4Material.hh"
#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"

// npy-
#include "NPY.hpp"
#include "NLog.hpp"


const char* CTraverser::GROUPVEL = "GROUPVEL" ; 


void CTraverser::init()
{
    m_ltransforms = NPY<float>::make(0, 4, 4);
    m_gtransforms = NPY<float>::make(0, 4, 4);
}


void CTraverser::Traverse()
{
     if(!m_top) 
     {
        LOG(fatal) << "CTraverser::Traverse m_top NULL" ;
        return ;
     }

     G4LogicalVolume* lv = m_top->GetLogicalVolume() ;
     TraverseVolumeTree(lv, 0 );


     m_pvnames.clear();
     std::vector<const G4VPhysicalVolume*> ancestors ; 
     AncestorTraverse(ancestors, m_top);
}

void CTraverser::AncestorTraverse(std::vector<const G4VPhysicalVolume*> ancestors, const G4VPhysicalVolume* pv)
{
     ancestors.push_back(pv); 
     AncestorVisit(ancestors);

     G4LogicalVolume* lv = pv->GetLogicalVolume() ;
     for (int i=0 ; i<lv->GetNoDaughters() ;i++) AncestorTraverse(ancestors, lv->GetDaughter(i) ); 
}

void CTraverser::AncestorVisit(std::vector<const G4VPhysicalVolume*> ancestors)
{
    G4Transform3D T ; 

    for(unsigned int i=0 ; i < ancestors.size() ; i++)
    {
        const G4VPhysicalVolume* apv = ancestors[i] ;

        G4RotationMatrix rot, invrot;
        if (apv->GetFrameRotation() != 0)
        {   
            rot = *(apv->GetFrameRotation());
            invrot = rot.inverse();
        }

        G4Transform3D P(invrot,apv->GetObjectTranslation()); 

        T = T*P ; 
    }
    const G4VPhysicalVolume* pv = ancestors.back() ; 

    LOG(debug) << "CTraverser::AncestorVisit " 
              << " size " << std::setw(3) << ancestors.size() 
              << " gcount " << std::setw(6) << m_gcount 
              << " pvname " << pv->GetName() 
              ;
    m_gcount += 1 ; 

    collectTransformT(m_gtransforms, T );
    m_pvnames.push_back(pv->GetName());
}

glm::mat4 CTraverser::getGlobalTransform(unsigned int index)
{
    return m_gtransforms->getMat4(index);
}
glm::mat4 CTraverser::getLocalTransform(unsigned int index)
{
    return m_ltransforms->getMat4(index);
}

unsigned int CTraverser::getNumGlobalTransforms()
{
    return m_gtransforms->getShape(0);
}
unsigned int CTraverser::getNumLocalTransforms()
{
    return m_ltransforms->getShape(0);
}



const char* CTraverser::getPVName(unsigned int index)
{
    return m_pvnames[index].c_str();
}



G4Transform3D CTraverser::TraverseVolumeTree(const G4LogicalVolume* const lv, const G4int depth)
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

         daughterR = TraverseVolumeTree(physvol->GetLogicalVolume(),depth+1); 

         // G4Transform3D P(rot,physvol->GetObjectTranslation());  GDML does this : not inverting the rotation portion 
         G4Transform3D P(invrot,physvol->GetObjectTranslation());

         VisitPV(physvol, invR*P*daughterR);

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

     addMaterial(material);

     G4MaterialPropertiesTable* mpt = material->GetMaterialPropertiesTable();
     if(mpt == NULL)
         addMaterialWithoutMPT(material);


     return R ;  
}

void CTraverser::VisitPV(const G4VPhysicalVolume* const pv, const G4Transform3D& T )
{
    LOG(debug) << "CTraverser::VisitPV" 
              << " lcount " << std::setw(6) << m_lcount 
              << " pvname " << pv->GetName() 
              ;
    m_lcount += 1 ; 

    collectTransformT(m_ltransforms, T );
}

void CTraverser::collectTransform(NPY<float>* buffer, const G4Transform3D& T)
{
    unsigned int nval = 16 ; 
    float* vals = new float[nval]; 

    vals[0] = T.xx() ; 
    vals[1] = T.xy() ; 
    vals[2] = T.xz() ; 
    vals[3] = T.dx() ; 

    vals[4] = T.yx() ; 
    vals[5] = T.yy() ; 
    vals[6] = T.yz() ; 
    vals[7] = T.dy() ; 

    vals[8] = T.zx() ; 
    vals[9] = T.zy() ; 
    vals[10] = T.zz() ; 
    vals[11] = T.dz() ; 

    vals[12] = 0 ; 
    vals[13] = 0 ; 
    vals[14] = 0 ; 
    vals[15] = 1 ; 

    buffer->add(vals, nval) ;

    delete vals ; 
}


void CTraverser::collectTransformT(NPY<float>* buffer, const G4Transform3D& T)
{
    unsigned int nval = 16 ; 
    float* vals = new float[nval]; 

    vals[0] = T.xx() ; 
    vals[1] = T.yx() ; 
    vals[2] = T.zx() ; 
    vals[3] = 0 ; 

    vals[4] = T.xy() ; 
    vals[5] = T.yy() ; 
    vals[6] = T.zy() ; 
    vals[7] = 0 ; 

    vals[8] = T.xz() ; 
    vals[9] = T.yz() ; 
    vals[10] = T.zz() ; 
    vals[11] = 0 ; 

    vals[12] = T.dx() ; 
    vals[13] = T.dy() ; 
    vals[14] = T.dz() ; 
    vals[15] = 1 ; 

    buffer->add(vals, nval) ;

    delete vals ; 
}


void CTraverser::saveTransforms(const char* path)
{
    //m_ltransforms->save(path);
    m_gtransforms->save(path);
}


void CTraverser::Visit(const G4LogicalVolume* const lv)
{
    const G4String lvname = lv->GetName();
    G4VSolid* solid = lv->GetSolid();
    G4Material* material = lv->GetMaterial();
     
    const G4String geoname = solid->GetName() ;
    const G4String matname = material->GetName();

    if(m_verbosity > 1 )
        LOG(info) << "CTraverser::Visit"
               << std::setw(20) << lvname
               << std::setw(50) << geoname
               << std::setw(20) << matname
               ;
}


bool CTraverser::hasMaterial(const G4Material* material)
{
     return std::find(m_materials.begin(), m_materials.end(), material) != m_materials.end()  ;
}

bool CTraverser::hasMaterialWithoutMPT(G4Material* material)
{
     return std::find(m_materials_without_mpt.begin(), m_materials_without_mpt.end(), material) != m_materials_without_mpt.end()  ;
}

void CTraverser::addMaterial(const G4Material* material)
{
     if(!hasMaterial(material)) 
     {
         m_materials.push_back(material) ;
     }
}

void CTraverser::addMaterialWithoutMPT(G4Material* material)
{
     if(!hasMaterialWithoutMPT(material)) 
     {
         m_materials_without_mpt.push_back(material) ;
     }
}



void CTraverser::dumpMaterials(const char* msg)
{
    LOG(info) << msg ; 
    for(unsigned int i=0 ; i < m_materials.size() ; i++)
    {
        const G4Material* material = m_materials[i];
        dumpMaterial(material);
    } 
}


void CTraverser::createGroupVel()
{
    // First get of GROUPVEL property creates it 
    // based on RINDEX property

    for(unsigned int i=0 ; i < m_materials.size() ; i++)
    {
        const G4Material* material = m_materials[i];
        G4MaterialPropertiesTable* mpt = material->GetMaterialPropertiesTable();
        if(mpt)
        {
            G4MaterialPropertyVector* gv = mpt->GetProperty(GROUPVEL);  
            unsigned int len = gv->GetVectorLength() ;
            if(m_verbosity > 1 )
                 LOG(info) << "CTraverser::createGroupVel" 
                           << " material " << material->GetName()
                           << " groupvel len " << len
                       ;
        }
        else
        {
            LOG(warning) << "CTraverser::createGroupVel"
                         << " material lacks MPT " << i << " " << material->GetName() ;
        } 
    } 
}

void CTraverser::Summary(const char* msg)
{
    LOG(info) << msg 
              << " numMaterials " << getNumMaterials() 
              << " numMaterialsWithoutMPT " << getNumMaterialsWithoutMPT() 
              ;
}

void CTraverser::dumpMaterial(const G4Material* material)
{
    LOG(info) << material->GetName() ;
    G4MaterialPropertiesTable* mpt = material->GetMaterialPropertiesTable();
    if(!mpt) return ;

    typedef std::map<G4String,G4MaterialPropertyVector*,std::less<G4String> >  PMAP ;
    typedef std::map< G4String, G4double,std::less<G4String> > CMAP ; 

    const PMAP* pm = mpt->GetPropertiesMap();
    const CMAP* cm = mpt->GetPropertiesCMap(); 

     std::stringstream ss ; 
    for(PMAP::const_iterator it=pm->begin() ; it!=pm->end() ; it++)
    {
        G4String pname = it->first ; 
        G4MaterialPropertyVector* pvec = it->second ; 
        ss << pname << " " ;  
        if(m_verbosity > 1 )
        dumpMaterialProperty(pname, pvec);
    }    
    std::string props = ss.str() ;

    LOG(info) <<  props ; 
}


void CTraverser::dumpMaterialProperty(const G4String& name, const G4MaterialPropertyVector* pvec)
{
    unsigned int len = pvec->GetVectorLength() ;

    LOG(info) << name 
              << " len " << len 
              << " h_Planck*c_light/nm " << h_Planck*c_light/nm
              ;

    for (unsigned int i=0; i<len; i++)
    {   
        G4double energy = pvec->Energy(i) ;
        G4double wavelength = h_Planck*c_light/energy ;
        G4double val = (*pvec)[i] ;

        LOG(info)
                  << std::fixed << std::setprecision(3) 
                  << " eV " << std::setw(10) << energy/eV
                  << " nm " << std::setw(10) << wavelength/nm
                  << " v  " << std::setw(10) << val ;
    }   
}





