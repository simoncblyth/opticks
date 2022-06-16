#include <string>
#include <vector>
#include <sstream>

#include "SPath.hh"
#include "SDir.h"
#include "SStr.hh"
#include "SSim.hh"
#include "SBnd.h"
#include "sdomain.h"
#include "NPFold.h"
#include "NP.hh"
#include "PLOG.hh"

#include "G4Material.hh"
#include "U4Material.hh"
#include "U4MaterialPropertyVector.h"

const plog::Severity U4Material::LEVEL = PLOG::EnvLevel("U4Material", "DEBUG"); 

std::string U4Material::DescMaterialTable()
{
    std::vector<std::string> names ; 
    GetMaterialNames(names);  
    std::stringstream ss ; 
    ss << "U4Material::DescMaterialTable " << names.size() << std::endl ; 
    for(int i=0 ; i < int(names.size()) ; i++) ss << names[i] << std::endl ; 
    std::string s = ss.str(); 
    return s ; 
}

void U4Material::GetMaterialNames( std::vector<std::string>& names)
{
    G4MaterialTable* tab =  G4Material::GetMaterialTable(); 
    for(int i=0 ; i < int(tab->size()) ; i++)
    {
        G4Material* mat = (*tab)[i] ; 
        const G4String& name = mat->GetName(); 
        names.push_back(name); 
    }
}


G4Material* U4Material::Get(const char* name)
{
   G4Material* material = G4Material::GetMaterial(name); 
   if( material == nullptr )
   {   
       material = Get_(name); 
   }   
   return material ;   
}


G4Material* U4Material::Get_(const char* name)
{
   G4Material* material = nullptr ; 
   if(strcmp(name, "Vacuum")==0)  material = Vacuum(name); 
   return material ; 
}

G4Material* U4Material::Vacuum(const char* name)
{
    G4double z, a, density ;
    G4Material* material = new G4Material(name, z=1., a=1.01*CLHEP::g/CLHEP::mole, density=CLHEP::universe_mean_density );
    return material ;
}

G4MaterialPropertyVector* U4Material::GetProperty(const G4Material* mat, const char* name)
{
    G4MaterialPropertiesTable* mpt = mat->GetMaterialPropertiesTable();
    return mpt->GetProperty(name) ; 
}

void U4Material::RemoveProperty( const char* key, G4Material* mat )
{
    if(mat == nullptr) return ; 
    const G4String& name = mat->GetName(); 
    G4MaterialPropertiesTable* mpt = mat->GetMaterialPropertiesTable();
    G4MaterialPropertyVector* prop = mpt->GetProperty(key); 

    if( prop == nullptr)
    {
         LOG(info) << " material " << name << " does not have property " << key ; 
    }
    else
    {
         LOG(info) << " material " << name << " removing property " << key ; 
         mpt->RemoveProperty(key);
         prop = mpt->GetProperty(key); 
         assert( prop == nullptr ); 
    }
}


void U4Material::GetPropertyNames( std::vector<std::string>& names, const G4Material* mat )
{
    G4MaterialPropertiesTable* mpt = mat->GetMaterialPropertiesTable();
    const G4MaterialPropertiesTable* mpt_ = mat->GetMaterialPropertiesTable();

    std::vector<G4String> pnames = mpt_->GetMaterialPropertyNames();

    typedef std::map<G4int, G4MaterialPropertyVector*, std::less<G4int> > MIV ; 
    const MIV* miv =  mpt->GetPropertyMap(); 
    for(MIV::const_iterator it=miv->begin() ; it != miv->end() ; it++ )
    {
         G4String name = pnames[it->first] ;  
         names.push_back(name.c_str()) ; 
    }
}


NPFold* U4Material::GetPropertyFold()
{
    NPFold* fold = new NPFold ; 

    std::vector<std::string> matnames ; 
    GetMaterialNames(matnames); 
    for(unsigned i=0 ; i < matnames.size() ; i++)
    { 
        const char* material = matnames[i].c_str(); 
        const G4Material* mat = G4Material::GetMaterial(material) ; 

        std::vector<std::string> propnames ; 
        GetPropertyNames( propnames, mat ); 

        for(unsigned j=0 ; j < propnames.size() ; j++)
        {
            const char* propname = propnames[j].c_str() ; 
            G4MaterialPropertyVector* prop = GetProperty(mat, propname );  
            NP* a = U4MaterialPropertyVector::ConvertToArray(prop); 
            fold->add( SStr::Format("%s/%s", material, propname), a ); 
        }
    }  
    return fold ; 
}


NPFold* U4Material::GetPropertyFold(const G4Material* mat )
{
    NPFold* fold = new NPFold ; 
    std::vector<std::string> propnames ; 
    GetPropertyNames( propnames, mat ); 
    for(unsigned i=0 ; i < propnames.size() ; i++)
    {
        const char* propname = propnames[i].c_str() ; 
        G4MaterialPropertyVector* prop = GetProperty(mat, propname );  
        NP* a = U4MaterialPropertyVector::ConvertToArray(prop); 
        fold->add( propname, a ); 
    }
    return fold ; 
}

 




std::string U4Material::DescPropertyNames( const G4Material* mat )
{
    G4MaterialPropertiesTable* mpt = mat->GetMaterialPropertiesTable();
    const G4MaterialPropertiesTable* mpt_ = mat->GetMaterialPropertiesTable();
    std::vector<G4String> pnames = mpt_->GetMaterialPropertyNames();
    std::vector<G4String> cnames = mpt_->GetMaterialConstPropertyNames();
    std::stringstream ss ; 
    ss << "U4Material::DescPropertyNames " << mat->GetName() << std::endl ; 

    /*
    ss << " MaterialPropertyNames.size " << pnames.size() << std::endl ; 
    for(int i=0 ; i < int(pnames.size()) ; i++) ss << pnames[i] << std::endl ; 

    ss << " MaterialConstPropertyNames.size " << cnames.size() << std::endl ; 
    for(int i=0 ; i < int(cnames.size()) ; i++) ss << cnames[i] << std::endl ; 
    */

    /*
    ss << " GetPropertiesMap " << std::endl ; 
    typedef std::map< G4String, G4MaterialPropertyVector*, std::less<G4String> > MSV ; 
    MSV* msv = mpt->GetPropertiesMap() ; 
    for(MSV::const_iterator it=msv->begin() ; it != msv->end() ; it++ ) ss << it->first << std::endl ; 

    ss << " GetPropertiesCMap " << std::endl ; 
    typedef std::map< G4String, G4double, std::less<G4String> > MSC ;
    MSC* msc = mpt->GetPropertiesCMap();
    for(MSC::const_iterator it=msc->begin() ; it != msc->end() ; it++ ) ss << it->first << std::endl ; 
    */

    ss << " GetPropertyMap " << std::endl ; 
    typedef std::map<G4int, G4MaterialPropertyVector*, std::less<G4int> > MIV ; 
    const MIV* miv =  mpt->GetPropertyMap(); 
    for(MIV::const_iterator it=miv->begin() ; it != miv->end() ; it++ ) ss << pnames[it->first] << std::endl ; 

    ss << " GetConstPropertyMap " << std::endl ; 
    typedef std::map<G4int, G4double, std::less<G4int> > MIC ; 
    const MIC* mic =  mpt->GetConstPropertyMap(); 
    for(MIC::const_iterator it=mic->begin() ; it != mic->end() ; it++ ) ss << cnames[it->first] << std::endl ; 


    std::string s = ss.str(); 
    return s ; 
}





/**
U4Material::MakeMaterialPropertiesTable
-----------------------------------------

The matdir may use the standard tokens eg::

   $IDPath/GScintillatorLib/LS_ori
   $IDPath/GMaterialLib/Water_ori

**/

G4MaterialPropertiesTable* U4Material::MakeMaterialPropertiesTable(const char* matdir_ )
{
    const char* matdir = SPath::Resolve(matdir_, NOOP); 
    std::vector<std::string> names ; 
    SDir::List(names, matdir, ".npy" ); 

    G4MaterialPropertiesTable* mpt = new G4MaterialPropertiesTable();

    std::stringstream ss ; 
    ss << "matdir " << matdir << " names " << names.size() << std::endl ; 

    for(unsigned i=0 ; i < names.size() ; i++)
    {
        const char* name = names[i].c_str(); 
        const char* key = SStr::HeadFirst(name, '.'); 

        NP* a = NP::Load(matdir, name  );         

        char type = Classify(a); 
        double* values = a->values<double>() ; 
        ss << Desc(key, a) << std::endl ;   
        
        switch(type)
        {
            case 'C': mpt->AddConstProperty(key, values[1])    ; break ; 
            case 'P': mpt->AddProperty(key, MakeProperty(a))   ; break ; 
            case 'F': mpt->AddProperty(key, MakeProperty(a))   ; break ; 
        }
    }

    std::string s = ss.str(); 
    
    LOG(LEVEL) << s ; 
    //std::cout << s << std::endl ; 
    
    return mpt ; 
} 


/**
U4Material::MakeMaterialPropertiesTable
----------------------------------------

The matdir may use the standard tokens eg::

   $IDPath/GScintillatorLib/LS_ori
   $IDPath/GMaterialLib/Water_ori

Constructs the properties table by loading the  properties listed in *delim* delimited *keys_*. 

**/

G4MaterialPropertiesTable* U4Material::MakeMaterialPropertiesTable(const char* matdir_, const char* keys_, char delim )
{
    const char* matdir = SPath::Resolve(matdir_, NOOP); 

    std::vector<std::string> keys ; 
    SStr::Split( keys_, delim, keys ); 

    G4MaterialPropertiesTable* mpt = new G4MaterialPropertiesTable();
    for(unsigned i=0 ; i < keys.size() ; i++)
    {
        const std::string& key_ = keys[i]; 
        const char* key = key_.c_str(); 
        const char* name = SStr::Format("%s.npy", key );   
        NP* a = NP::Load(matdir, name );         
        assert(a); 

        LOG(info)
             << " key " << std::setw(15) << key 
             << " a.desc " << a->desc() 
             ; 

        G4MaterialPropertyVector* v = MakeProperty(a); 
        mpt->AddProperty(key, v);    
    }
    return mpt ; 
} 




G4MaterialPropertyVector* U4Material::MakeProperty(const NP* a)  // static
{
    std::vector<double> d, v ; 
    a->psplit<double>(d,v);   // split array into domain and values 
    assert(d.size() == v.size());
    //assert(d.size() > 1 );  // OpticalCONSTANT (scint time fraction property misusage) breaks this  
    G4MaterialPropertyVector* mpv = new G4MaterialPropertyVector(d.data(), v.data(), d.size() );  
    return mpv ; 
}






/**
U4Material::Classify
---------------------

Heuristic to distinguish ConstProperty from Property based on array size and content
and identify fractional property. 

C
   ConstProperty
P
   Property
F
   Property that looks like a fractional split with a small number of values summing to 1.

**/

char U4Material::Classify(const NP* a)
{
    assert(a); 
    assert(a->shape.size() == 2); 
    assert(a->has_shape(-1,2) && a->ebyte == 8 && a->uifc == 'f' ); 
    const double* values = a->cvalues<double>() ; 
    char type = a->has_shape(2,2) && values[1] == values[3] ? 'C' : 'P' ; 

    int numval = a->shape[0]*a->shape[1] ; 
    double fractot = 0. ; 
    if(type == 'P' && numval <= 8) 
    {
        for(int i=0 ; i < numval ; i++) if(i%2==1) fractot += values[i] ;   
        if( fractot == 1. ) type = 'F' ;  
    }  
    return type ; 
}


std::string U4Material::Desc(const char* key, const NP* a )
{
    char type = Classify(a); 
    const double* values = a->cvalues<double>() ; 
    int numval = a->shape[0]*a->shape[1] ; 
    double value = type == 'C' ? values[1] : 0. ; 

    std::stringstream ss ; 

    ss << std::setw(20) << key 
       << " " << std::setw(10) << a->sstr() 
       << " type " << type
       << " value " << std::setw(10) << std::fixed << std::setprecision(3) << value 
       << " : " 
       ; 

    double tot = 0. ; 
    if(numval <= 8) 
    {
        for(int i=0 ; i < numval ; i++) if(i%2==1) tot += values[i] ;   
        for(int i=0 ; i < numval ; i++) ss << std::setw(10) << std::fixed << std::setprecision(3) << values[i] << " " ;
        ss << " tot: " << std::setw(10) << std::fixed << std::setprecision(3) << tot << " " ; 
    } 
    std::string s = ss.str(); 
    return s ; 
}






G4MaterialPropertiesTable*  U4Material::MakeMaterialPropertiesTable_FromProp( 
     const char* a_key, const G4MaterialPropertyVector* a_prop,
     const char* b_key, const G4MaterialPropertyVector* b_prop,
     const char* c_key, const G4MaterialPropertyVector* c_prop,
     const char* d_key, const G4MaterialPropertyVector* d_prop
)
{
    G4MaterialPropertiesTable* mpt = new G4MaterialPropertiesTable();
    if(a_key && a_prop) mpt->AddProperty(a_key,const_cast<G4MaterialPropertyVector*>(a_prop));    //  HUH: why not const ?
    if(b_key && b_prop) mpt->AddProperty(b_key,const_cast<G4MaterialPropertyVector*>(b_prop));
    if(c_key && c_prop) mpt->AddProperty(c_key,const_cast<G4MaterialPropertyVector*>(c_prop));
    if(d_key && d_prop) mpt->AddProperty(d_key,const_cast<G4MaterialPropertyVector*>(d_prop));
    return mpt ;
}




G4Material* U4Material::MakeWater(const char* name)  // static
{
    G4double a, z, density;
    G4int nelements;
    G4Element* O = new G4Element("Oxygen"  , "O", z=8 , a=16.00*CLHEP::g/CLHEP::mole);
    G4Element* H = new G4Element("Hydrogen", "H", z=1 , a=1.01*CLHEP::g/CLHEP::mole);
    G4Material* mat = new G4Material(name, density= 1.0*CLHEP::g/CLHEP::cm3, nelements=2);
    mat->AddElement(H, 2);
    mat->AddElement(O, 1);
    return mat ; 
}


G4Material* U4Material::MakeMaterial(const G4MaterialPropertyVector* rindex, const char* name)  // static
{
    G4Material* mat = MakeWater(name); 
    G4MaterialPropertiesTable* mpt = MakeMaterialPropertiesTable_FromProp("RINDEX", rindex) ;
    mat->SetMaterialPropertiesTable(mpt) ;
    return mat ;
}

/**
U4Material::MakeMaterial
--------------------------

::

    G4Material* mat = MakeMaterial("LS", "$IDPath/GScintillatorLib/LS_ori", "RINDEX,FASTCOMPONENT,SLOWCOMPONENT,REEMISSIONPROB,GammaCONSTANT,OpticalCONSTANT");  

**/


G4Material* U4Material::MakeMaterial(const char* name, const char* matdir, const char* props )
{
    G4Material* mat = MakeWater(name); 
    G4MaterialPropertiesTable* mpt = MakeMaterialPropertiesTable(matdir, props, ','); 
    mat->SetMaterialPropertiesTable(mpt) ;
    return mat ;
}

G4Material* U4Material::MakeMaterial(const char* name, const char* matdir)
{
    G4Material* mat = MakeWater(name); 
    G4MaterialPropertiesTable* mpt = MakeMaterialPropertiesTable(matdir); 
    mat->SetMaterialPropertiesTable(mpt) ;
    return mat ;
}




/**
U4Material::MakeScintillator
---------------------------------

Creates material with Water G4Element and density and then loads the properties of LS into its MPT 

**/

G4Material* U4Material::MakeScintillator()
{
    G4Material* mat = MakeMaterial("LS", "$IDPath/GScintillatorLib/LS_ori");  
    return mat ; 
}


/**
U4Material::LoadOri
---------------------

Currently original material properties are not standardly persisted, 
so typically will have to override the IDPath envvar for this to 
succeed to load. 

**/

G4Material* U4Material::LoadOri(const char* name)
{
    const char* matdir = SStr::Format("%s/%s_ori", LIBDIR, name );   
    G4Material* mat = MakeMaterial(name, matdir );  
    return mat ; 
}

void U4Material::ListOri(std::vector<std::string>& names)
{
    const char* libdir = SPath::Resolve(LIBDIR, NOOP ); 
    const char* ext = "_ori" ; 
    SDir::List(names, libdir, ext ); 
    SDir::Trim(names, ext); 
    LOG(LEVEL) 
        << " libdir " << libdir
        << " ext " << ext 
        << " names.size " << names.size() 
        ; 
}

void U4Material::LoadOri()
{
    size_t num_mat[2] ; 
    num_mat[0] = G4Material::GetNumberOfMaterials(); 
    std::vector<std::string> names ; 
    ListOri(names);  

    for(int i=0 ; i < int(names.size()) ; i++)
    {
        const std::string& name = names[i] ; 
        const char* n = name.c_str(); 
        LOG(LEVEL) << n ; 
        G4Material* mat = LoadOri(n); 
        const G4String& name_ = mat->GetName(); 
        assert( strcmp(name_.c_str(), n ) == 0 ); 

    }
    num_mat[1] = G4Material::GetNumberOfMaterials(); 
    LOG(info) << "Increased G4Material::GetNumberOfMaterials from " << num_mat[0] << " to " << num_mat[1] ;  

}



/**
U4Material::LoadBnd
--------------------

HMM: if the material exists already then need to change its 
properties, not scrub the pre-existing material. 
Thats needed for scintillators as the standard bnd properties
do not include all the needed scint props. 

Load the material properties from the SSim::get_bnd array using SBnd::getPropertyGroup 
for each material. 

**/

void U4Material::LoadBnd()
{
    SSim* sim = SSim::Load(); 
    const SBnd* sb = sim->get_sbnd(); 

    std::vector<std::string> names ; 
    sb->getMaterialNames(names );

    for(unsigned mat=0 ; mat < names.size() ; mat++) 
    {
        const char* name = names[mat].c_str() ; 
        G4Material* prior = G4Material::GetMaterial(name);  
        if(prior != nullptr) std::cout << "prior material " << name << std::endl ; 

        NP* pg = sb->getPropertyGroup(name,-1); 

        int ni = 2 ;                              // groups of quad props
        int nj = sdomain::FINE_DOMAIN_LENGTH ;    // wavelength domain
        int nk = 4 ;                              // 4 props

        assert( pg->has_shape(ni, nj, nk));  
        std::cout << pg->desc() << std::endl ; 
        assert( pg->ebyte == 8 );  // HUH ? 

        double* vv = pg->values<double>(); 
        for(int i=0 ; i < ni ; i++)  
        {
            for(int j=0 ; j < nj ; j++)
            {
                for(int k=0 ; k < 4 ; k++)
                {
                    int idx = nk*nj*i+nk*j+k ; 
                    double v = vv[idx] ; 
                    
                    if( i == 1 && k == 0 ) std::cout 
                        << std::setw(6) << idx 
                        << " : " 
                        << std::setw(10) << std::fixed << std::setprecision(5) << v 
                        << std::endl ; 

                }
            }
        }


    }
}


