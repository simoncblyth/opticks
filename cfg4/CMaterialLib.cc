
#include "G4Material.hh"
#include "G4MaterialPropertiesTable.hh"
#include "globals.hh"
#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"


#include "NPY.hpp"

#include "Opticks.hh"
#include "OpticksHub.hh"

#include "GMaterial.hh"
#include "GMaterialLib.hh"
#include "GBndLib.hh"

#include "CMPT.hh"
#include "CMaterialLib.hh"

#include "PLOG.hh"


CMaterialLib::CMaterialLib(OpticksHub* hub) 
   :
   CPropLib(hub, 0),
   m_converted(false)
{
}


void CMaterialLib::convert()
{
    assert(m_converted == false);
    m_converted = true ; 

    unsigned int ngg = getNumMaterials() ;
    for(unsigned int i=0 ; i < ngg ; i++)
    {
        const GMaterial* ggmat = getMaterial(i);
        const char* name = ggmat->getShortName() ;
        const G4Material* g4mat = convertMaterial(ggmat);

        // special cased GROUPVEL getter invokes setGROUPVEL which adds the property to the MPT 
        // derived from RINDEX
        G4MaterialPropertyVector* groupvel = g4mat->GetMaterialPropertiesTable()->GetProperty("GROUPVEL") ;
        assert(groupvel);

        std::string keys = getMaterialKeys(g4mat);
        LOG(debug) << "CMaterialLib::convert : converted ggeo material to G4 material " << name << " with keys " << keys ;  
    }
    LOG(info) << "CMaterialLib::convert : converted " << ngg << " ggeo materials to G4 materials " ; 
}




const G4Material* CMaterialLib::makeMaterial(const char* matname)
{
     GMaterial* kmat = m_mlib->getMaterial(matname) ;
     const G4Material* material = convertMaterial(kmat);
     return material ; 
}


const G4Material* CMaterialLib::convertMaterial(const GMaterial* kmat)
{
    if(!kmat)
    {
        LOG(fatal) << "CMaterialLib::convertMaterial NULL kmat " ;
    } 
    assert(kmat);

    const char* name = kmat->getShortName();
    const G4Material* prior = getG4Material(name) ;
    if(prior)
    {
        LOG(info) << "CMaterialLib::convertMaterial" 
                  << " REUSING PRIOR G4Material "
                  << " name " << std::setw(35) << name
                  << " prior " << (void*)prior 
                  ;
 
        return prior ; 
    }


    unsigned int materialIndex = m_mlib->getMaterialIndex(kmat);

    G4String sname = name ; 

    LOG(debug) << "CMaterialLib::convertMaterial  " 
              << " name " << name
              << " sname " << sname
              << " materialIndex " << materialIndex
              ;

    G4Material* material(NULL);
    if(strcmp(name,"MainH2OHale")==0)
    {
        material = makeWater(name) ;
    } 
    else
    {
        G4double z, a, density ;
        // presumably z, a and density are not relevant for optical photons 
        material = new G4Material(sname, z=1., a=1.01*g/mole, density=universe_mean_density );
    }


    LOG(trace) << "." ; 

    G4MaterialPropertiesTable* mpt = makeMaterialPropertiesTable(kmat);
    material->SetMaterialPropertiesTable(mpt);

    m_ggtog4[kmat] = material ; 
    m_g4mat[name] = material ;   // used by getG4Material(shortname) 

    return material ;  
}






const G4Material* CMaterialLib::getG4Material(const char* shortname)
{
    const G4Material* mat =  m_g4mat.count(shortname) == 1 ? m_g4mat[shortname] : NULL ; 
    return mat ; 
}





G4Material* CMaterialLib::makeWater(const char* name)
{
    G4double z,a;
    G4Element* H  = new G4Element("Hydrogen" ,"H" , z= 1., a=   1.01*g/mole);
    G4Element* O  = new G4Element("Oxygen"   ,"O" , z= 8., a=  16.00*g/mole);

    G4double density;
    G4int ncomponents, natoms;

    G4Material* material = new G4Material(name, density= 1.000*g/cm3, ncomponents=2);
    material->AddElement(H, natoms=2);
    material->AddElement(O, natoms=1);

    return material ; 
}

G4Material* CMaterialLib::makeVacuum(const char* name)
{
    G4double z, a, density ;

    // Vacuum standard definition...
    G4Material* material = new G4Material(name, z=1., a=1.01*g/mole, density=universe_mean_density );

    return material ; 
}











void CMaterialLib::dump(const char* msg)
{
    LOG(info) <<  msg ; 

    unsigned int ni = getNumMaterials() ;
    int index = m_ok->getLastArgInt();
    const char* lastarg = m_ok->getLastArg();


    LOG(trace) <<  " ni " << ni 
               << " index " << index
               << " lastarg " << lastarg
              ; 


    if(index < int(ni) && index >=0)
    {   
        LOG(trace) << " dump index " << index ;
        const GMaterial* mat = getMaterial(index);
        dump(mat, msg);
    }   
    else if(hasMaterial(lastarg))
    {   
        LOG(trace) << " dump lastarg " << lastarg ;
        const GMaterial* mat = getMaterial(lastarg);
        dump(mat, msg);
    }   
    else
    {
        LOG(trace) << " dump ni " << ni  ;
        for(unsigned int i=0 ; i < ni ; i++)
        {
           const GMaterial* mat = getMaterial(i);
           dump(mat, msg);
        }
    }
}




void CMaterialLib::dump(const GMaterial* mat, const char* msg)
{
    LOG(trace) << " dump mat " << mat ;
    GMaterial* _mat = const_cast<GMaterial*>(mat); 
    const char* _name = _mat->getName();
    LOG(trace) << " dump _name " << _name ;
    const G4Material* g4mat = getG4Material(_name);
    LOG(trace) << " dump g4mat " << g4mat ;
    dumpMaterial(g4mat, msg);
}


void CMaterialLib::dumpMaterial(const G4Material* mat, const char* msg)
{

    if(mat == NULL)
    {
         LOG(error) << " NULL G4Material mat " << msg ; 
         return ; 
    }


    const G4String& name = mat->GetName();
    LOG(info) << msg << " name " << name ; 

    G4MaterialPropertiesTable* mpt = mat->GetMaterialPropertiesTable();
    //mpt->DumpTable();

    GPropertyMap<float>* pmap = convertTable( mpt , name );  // back into GGeo language for dumping purpose only

    unsigned int fw = 20 ;  
    bool dreciprocal = true ; 

    std::cout << pmap->make_table(fw, m_dscale, dreciprocal) << std::endl ;


    CMPT cmpt(mpt);
    cmpt.dumpRaw("RINDEX,GROUPVEL");

}


NPY<float>* CMaterialLib::makeArray(const char* name, const char* keys, bool reverse)
{
    const G4Material* mat = getG4Material(name);
    G4MaterialPropertiesTable* mpt = mat->GetMaterialPropertiesTable();
    CMPT cmpt(mpt);
    return cmpt.makeArray(keys, reverse);
}





void CMaterialLib::dumpMaterials(const char* msg)
{
    unsigned int ngg = getNumMaterials() ;
    LOG(info) << msg 
              << " numMaterials " << ngg
              ;

    for(unsigned int i=0 ; i < ngg ; i++)
    {
        const GMaterial* ggm = getMaterial(i);
        LOG(info) << "CMaterialLib::dumpMaterials" 
                  << " ggm (shortName) " << ggm->getShortName() 
                  ;
    }

    LOG(info) << msg  << " ggtog4" ; 
    typedef std::map<const GMaterial*, const G4Material*> MMM ; 
    for(MMM::const_iterator it=m_ggtog4.begin() ; it != m_ggtog4.end() ; it++)
    {
        const GMaterial* ggmat = it->first ; 
        const G4Material* g4mat = it->second ; 

        const G4String& name = g4mat->GetName();
        const char* ggname = ggmat->getShortName();

        std::cout << std::setw(40) << name 
                  << std::setw(40) << ggname 
                  << std::endl ; 

        dumpMaterial(g4mat, "g4mat");


    }
}



