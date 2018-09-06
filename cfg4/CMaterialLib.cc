#include "BStr.hh"

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
#include "CVec.hh"
#include "CMaterialLib.hh"

#include "PLOG.hh"


CMaterialLib::CMaterialLib(OpticksHub* hub) 
    :
    CPropLib(hub, 0),
    m_converted(false)
{
}

void CMaterialLib::dumpGroupvelMaterial(const char* msg, float wavelength, float groupvel, float tdiff, int step_id, const char* qwn)
{
    if(std::abs(wavelength - 430.f) < 0.01 )
    {
         std::string mat = firstMaterialWithGroupvelAt430nm( groupvel, 0.001f );
         LOG(info) 
                   << std::setw(3) << step_id
                   << std::setw(20) << msg 
                   << " nm " << std::setw(5) << wavelength 
                   << " nm/ns " << std::setw(10) << groupvel
                   << " ns " << std::setw(10) << tdiff 
                   << " lkp " << mat
                   << " qwn " << qwn 
                   ; 
    }
}

void CMaterialLib::fillMaterialValueMap()
{
    const char* matnames = "GdDopedLS,Acrylic,LiquidScintillator,MineralOil,Bialkali" ;

   // Bialkali for debug injection black ops 
    fillMaterialValueMap(m_groupvel_430nm, matnames, "GROUPVEL", 430.f );
    dumpMaterialValueMap(matnames, m_groupvel_430nm);
}

/*

See ana/bnd.py::

    In [21]: dict(filter(lambda kv:kv[1]<299,zip(i1m.names,i1m.data[:,1,430-60,0])))
        Out[21]: 
        {'Acrylic': 192.77956,
         'Bialkali': 205.61897,
         'DeadWater': 217.83527,
         'GdDopedLS': 194.5192,
         'IwsWater': 217.83527,
         'LiquidScintillator': 194.5192,
         'MineralOil': 197.13411,
         'OwsWater': 217.83527,
         'Pyrex': 205.61897,
         'Teflon': 192.77956,
         'Water': 217.83527}
*/


std::string CMaterialLib::firstMaterialWithGroupvelAt430nm(float groupvel, float delta)
{
    return CMaterialLib::firstKeyForValue( groupvel , m_groupvel_430nm, delta); 
}

void CMaterialLib::postinitialize()
{
   // invoked from CGeometry::postinitialize
    fillMaterialValueMap(); 
}

bool CMaterialLib::isConverted()
{
    return m_converted ; 
}


/**
CMaterialLib::convert
----------------------

This never getting called during standard running, 
nope materials are converted one-by-one from CTestDetector::makeDetector

* TODO: check this


With CCerenkovGeneratorTest are invoking from main

**/

void CMaterialLib::convert()
{
   //  assert(0) ; 
    assert(m_converted == false);
    m_converted = true ; 

    unsigned int ngg = getNumMaterials() ;
    for(unsigned int i=0 ; i < ngg ; i++)
    {
        const GMaterial* ggmat = getMaterial(i);
        const char* name = ggmat->getShortName() ;
        const G4Material* g4mat = convertMaterial(ggmat);

        // special cased GROUPVEL getter invokes setGROUPVEL which adds the property to the MPT 
        // derived from RINDEX id the GROUPVEL property is not already present
        //G4MaterialPropertyVector* groupvel = g4mat->GetMaterialPropertiesTable()->GetProperty("GROUPVEL") ;
        //assert(groupvel);

        G4MaterialPropertyVector* rindex = g4mat->GetMaterialPropertiesTable()->GetProperty("RINDEX") ;
        assert(rindex);

        G4double Pmin = rindex->GetMinLowEdgeEnergy();
        G4double Pmax = rindex->GetMaxLowEdgeEnergy();

        G4double Wmin = h_Planck*c_light/Pmax ;
        G4double Wmax = h_Planck*c_light/Pmin ;

        std::string keys = getMaterialKeys(g4mat);
        assert( !keys.empty() ); 

        //LOG(info) << "converted ggeo material to G4 material " << name << " with keys " << keys ;  
        LOG(info) 
            << " g4mat " << (void*)g4mat
            << " name " << name 
            << " Pmin " << Pmin
            << " Pmax " << Pmax
            << " Wmin " << Wmin/nm
            << " Wmax " << Wmax/nm
            ;  
    }
    LOG(info) << "CMaterialLib::convert : converted " << ngg << " ggeo materials to G4 materials " ; 
}


void CMaterialLib::saveGROUPVEL(const char* base)
{
   // invoked by GROUPVELTest 
    unsigned int ngg = getNumMaterials() ;
    for(unsigned int i=0 ; i < ngg ; i++)
    {
        const GMaterial* ggmat = getMaterial(i);
        const char* name = ggmat->getShortName() ;
        NPY<float>* pa = makeArray(name,"RINDEX,GROUPVEL");
        pa->save(base, name, "saveGROUPVEL.npy");
    }
}


const G4Material* CMaterialLib::makeG4Material(const char* matname)
{
     GMaterial* kmat = m_mlib->getMaterial(matname) ;
     const G4Material* material = convertMaterial(kmat);
     LOG(info) << "CMaterialLib::makeMaterial" 
               << " matname " << matname 
               << " material " << (void*)material
               ;
       
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


    LOG(verbose) << "." ; 

    G4MaterialPropertiesTable* mpt = makeMaterialPropertiesTable(kmat);
    material->SetMaterialPropertiesTable(mpt);

    m_ggtog4[kmat] = material ; 
    m_g4mat[name] = material ;   // used by getG4Material(shortname) 

    return material ;  
}




bool CMaterialLib::hasG4Material(const char* shortname)
{
    return  m_g4mat.count(shortname) ; 
}

const G4Material* CMaterialLib::getG4Material(const char* shortname)
{
    const G4Material* mat =  m_g4mat.count(shortname) == 1 ? m_g4mat[shortname] : NULL ; 
    return mat ; 
}

const CMPT* CMaterialLib::getG4MPT(const char* shortname)
{
    const G4Material* mat = getG4Material(shortname);
    if(mat == NULL)
    {
         LOG(warning) << "CMaterialLib::getG4MPT"
                      << " no G4Material " << shortname
                      ;

    }


    G4MaterialPropertiesTable* mpt = mat ? mat->GetMaterialPropertiesTable() : NULL ;
    return mpt ? new CMPT(mpt) : NULL ;
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


    LOG(verbose) <<  " ni " << ni 
               << " index " << index
               << " lastarg " << lastarg
              ; 


    if(index < int(ni) && index >=0)
    {   
        LOG(verbose) << " dump index " << index ;
        const GMaterial* mat = getMaterial(index);
        dump(mat, msg);
    }   
    else if(hasMaterial(lastarg))
    {   
        LOG(verbose) << " dump lastarg " << lastarg ;
        const GMaterial* mat = getMaterial(lastarg);
        dump(mat, msg);
    }   
    else
    {
        LOG(verbose) << " dump ni " << ni  ;
        for(unsigned int i=0 ; i < ni ; i++)
        {
           const GMaterial* mat = getMaterial(i);
           dump(mat, msg);
        }
    }
}




void CMaterialLib::dump(const GMaterial* mat, const char* msg)
{
    LOG(verbose) << " dump mat " << mat ;
    GMaterial* _mat = const_cast<GMaterial*>(mat); 
    const char* _name = _mat->getName();
    LOG(verbose) << " dump _name " << _name ;
    const G4Material* g4mat = getG4Material(_name);
    LOG(verbose) << " dump g4mat " << g4mat ;
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


    //CMPT cmpt(mpt);
    //cmpt.dumpRaw("RINDEX,GROUPVEL");

}


NPY<float>* CMaterialLib::makeArray(const char* name, const char* keys, bool reverse)
{
    const G4Material* mat = getG4Material(name);
    G4MaterialPropertiesTable* mpt = mat->GetMaterialPropertiesTable();
    CMPT cmpt(mpt);
    return cmpt.makeArray(keys, reverse);
}



/*
   names="GdDopedLS,Acrylic,LiquidScintillator,MineralOil"
   keys="GROUPVEL"
*/

void CMaterialLib::fillMaterialValueMap(std::map<std::string,float>& vmp,  const char* _matnames, const char* key, float nm)
{
    std::vector<std::string> matnames ; 
    BStr::split(matnames, _matnames, ',' );

    unsigned nmat = matnames.size();

    for(unsigned i=0 ; i<nmat ; i++)
    {
         const char* name = matnames[i].c_str() ;

         if(m_mlib->hasMaterial(name))
         {
             if(!hasG4Material(name))
             {
                 makeG4Material(name);
             }

             const CMPT* cmpt = getG4MPT(name);
             assert(cmpt);
             CVec* vec = cmpt->getCVec(key); 
             assert(vec);
             vmp[name] = vec->getValue(nm);
         }
    }
}


void CMaterialLib::dumpMaterialValueMap(const char* msg, std::map<std::string,float>& vmp)
{

    LOG(info) << msg ;
    typedef std::map<std::string, float> SF ; 
    for(SF::iterator it=vmp.begin() ; it != vmp.end() ; it++)
    {
         std::cout
              << std::setw(20) << it->first 
              << std::setw(20) << it->second
              << std::endl ; 
    } 
}

std::string CMaterialLib::firstKeyForValue(float val, std::map<std::string,float>& vmp, float delta)
{
    std::string empty ; 
    typedef std::map<std::string, float> SF ; 
    for(SF::iterator it=vmp.begin() ; it != vmp.end() ; it++)
    {
        std::string key = it->first ; 
        float kval = it->second ; 
        if( std::abs(kval-val) < delta ) return key ;
    }
    return empty ;
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

