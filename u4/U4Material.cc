#include <string>
#include <vector>
#include <sstream>
#include <limits>

#include "SPath.hh"
#include "SDir.h"
#include "SStr.hh"
#include "SSim.hh"
#include "SBnd.h"
#include "sdomain.h"
#include "NPFold.h"
#include "NP.hh"
#include "SLOG.hh"

#include "G4Material.hh"
#include "G4SystemOfUnits.hh"
#include "U4Material.hh"
#include "U4MaterialPropertyVector.h"

const plog::Severity U4Material::LEVEL = SLOG::EnvLevel("U4Material", "DEBUG"); 


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

void U4Material::GetMaterialNames( std::vector<std::string>& names, bool extras )
{
    G4MaterialTable* tab =  G4Material::GetMaterialTable(); 
    for(int i=0 ; i < int(tab->size()) ; i++)
    {
        G4Material* mat = (*tab)[i] ; 
        const G4String& name = mat->GetName(); 
        names.push_back(name); 
    }
    if(extras)
    {
        names.push_back(U4Material::SCINTILLATOR); 
        names.push_back(U4Material::VACUUM); 
   }
}

/**
U4Material::FindMaterialName
------------------------------

Look for the names of any of the materials within the volname 
string and return the first material name found. 
If more than one names are matched this asserts : so be careful
to avoid unwanted material names within volume names. 

**/

const char* U4Material::FindMaterialName(const char* volname)
{
    std::vector<std::string> names ; 
    GetMaterialNames(names, true); 
    const char* matname = nullptr ;  
    unsigned count = 0 ; 
    for(unsigned i=0 ; i < names.size() ; i++)
    {
        const char* mat = names[i].c_str(); 
        if(strstr(volname, mat) != nullptr )
        {
            if(count == 0 ) matname = strdup(mat); 
            count+= 1 ; 
        }
    }
    if( count > 1 ) LOG(fatal) << " count " << count << " volname " << volname << " matname " << matname ;  
    assert( count < 2 ); 
    return matname ; 
}




/**
U4Material::Get
-----------------

Ordinary G4Material::GetMaterial has first chance, otherwise U4Material::Get_ 
may provide. 

**/

G4Material* U4Material::Get(const char* name)
{
   G4Material* material = G4Material::GetMaterial(name, false); 
   if( material == nullptr ) material = Get_(name); 
   return material ;   
}

G4Material* U4Material::Get_(const char* name)
{
    G4Material* material = nullptr ; 
    if(strcmp(name, SCINTILLATOR)==0) material = MakeScintillator(); 
    if(strcmp(name, VACUUM)==0)       material = Vacuum(name); 
    return material ; 
}

G4Material* U4Material::Vacuum(const char* name)
{
    bool dump = false ; 
    if(dump) std::cout << "[ U4Material::Vacuum " << std::endl ; 
    G4double z = 1. ; 
    G4double a = 1.01*CLHEP::g/CLHEP::mole ; 
    G4double density = 1.00001*CLHEP::universe_mean_density ;  // curious setting to 1. gives a warning 
    if(dump) std::cout << " U4Material::Vacuum density " << std::scientific << density << std::endl ; 
    G4Material* material = new G4Material(name, z, a, density );
    if(dump) std::cout << "] U4Material::Vacuum " << std::endl ; 
    return material ;
}

G4MaterialPropertyVector* U4Material::GetProperty(const G4Material* mat, const char* pname)
{
    G4MaterialPropertiesTable* mpt = mat->GetMaterialPropertiesTable();
    return mpt->GetProperty(pname) ; 
}

void U4Material::GetMinMaxValue( double& mn , double& mx, const G4MaterialPropertyVector* prop )
{
    size_t len = prop->GetVectorLength() ; 
    mn = std::numeric_limits<double>::max(); 
    mx = std::numeric_limits<double>::min(); 
    for(unsigned i=0 ; i < len ; i++)
    {
        double v = (*prop)[i] ; 
        if( mx < v ) mx = v ; 
        if( mn > v ) mn = v ; 
    }
}

std::string U4Material::DescProperty(const G4MaterialPropertyVector* prop)
{
    std::stringstream ss ; 
    if( prop == nullptr )
    {
        ss << " prop null " ; 
    }
    else
    {
        G4MaterialPropertyVector* prop_ = const_cast<G4MaterialPropertyVector*>(prop); 
        double mn, mx ; 
        GetMinMaxValue(mn, mx, prop); 
        ss 
           << std::setw(16) << "GetVectorLength" << std::setw(5) << prop->GetVectorLength()
           << std::setw(5) << "mn" << std::fixed << std::setw(10) << std::setprecision(5) << mn
           << std::setw(5) << "mx" << std::fixed << std::setw(10) << std::setprecision(5) << mx
           << std::setw(12) << "GetMaxValue" << std::fixed << std::setw(10) << std::setprecision(5) << prop_->GetMaxValue()
           << std::setw(12) << "GetMinValue" << std::fixed << std::setw(10) << std::setprecision(5) << prop_->GetMinValue()
           << std::setw(23) << "GetMinLowEdgeEnergy/eV" << std::fixed << std::setw(10) << std::setprecision(5) << prop_->GetMinLowEdgeEnergy()/eV
           << std::setw(23) << "GetMaxLowEdgeEnergy/eV" << std::fixed << std::setw(10) << std::setprecision(5) << prop_->GetMaxLowEdgeEnergy()/eV
           ; 
    }
    std::string s = ss.str(); 
    return s ; 
}

std::string U4Material::DescProperty(const char* mname, const char* pname)
{
    const G4MaterialPropertyVector* prop = GetProperty(mname, pname); 
    std::stringstream ss ; 
    ss << std::setw(20) << mname << std::setw(20) << pname ; 
    ss << DescProperty(prop) ; 
    std::string s = ss.str(); 
    return s ; 
}

std::string U4Material::DescProperty(const char* mname)
{
    G4Material* mat = G4Material::GetMaterial(mname, false); 
    std::vector<std::string> pnames ; 
    GetPropertyNames(pnames, mat);  

    std::stringstream ss ; 
    for(unsigned i=0 ; i < pnames.size() ; i++)  ss << DescProperty(mname, pnames[i].c_str() ) << std::endl ; 
    std::string s = ss.str(); 
    return s ; 
}

std::string U4Material::DescProperty()
{
    std::vector<std::string> mnames ; 
    GetMaterialNames(mnames);  
    std::stringstream ss ; 
    for(unsigned i=0 ; i < mnames.size() ; i++)  ss << DescProperty(mnames[i].c_str()) << std::endl ; 
    std::string s = ss.str(); 
    return s ; 
}





G4MaterialPropertyVector* U4Material::GetProperty(const char* mname, const char* pname)
{
    bool warning = false ; 
    const G4Material* mat = G4Material::GetMaterial(mname, warning);  
    return mat ? GetProperty(mat, pname) : nullptr ; 
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

    if( mpt_ == nullptr ) return ; 
 
    std::vector<G4String> pnames = mpt_->GetMaterialPropertyNames();

    typedef std::map<G4int, G4MaterialPropertyVector*, std::less<G4int> > MIV ; 
    const MIV* miv =  mpt->GetPropertyMap(); 
    for(MIV::const_iterator it=miv->begin() ; it != miv->end() ; it++ )
    {
         G4String name = pnames[it->first] ;  
         names.push_back(name.c_str()) ; 
    }
}

/**
U4Material::MakePropertyFold
----------------------------

Converts the properties of all materials into a 
NPFold with keys of form : "materialName"/"propName" 

HMM: perhaps NPFold can be made to handle sub-NPFold 
more formally than this, as clearly sometimes want to 
just load single materials and sometimes all materials. 

**/

NPFold* U4Material::MakePropertyFold_flat()
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


NPFold* U4Material::MakePropertyFold()
{
    NPFold* fold = new NPFold ; 

    std::vector<std::string> matnames ; 
    GetMaterialNames(matnames); 
    for(unsigned i=0 ; i < matnames.size() ; i++)
    { 
        const char* material = matnames[i].c_str(); 
        const G4Material* mat = G4Material::GetMaterial(material) ; 
        NPFold* matfold = MakePropertyFold(mat) ; 
        fold->add_subfold( material, matfold ); 
    }  
    return fold ; 
}


NPFold* U4Material::MakePropertyFold(std::vector<const G4Material*>& mats)
{
    NPFold* fold = new NPFold ; 
    for(unsigned i=0 ; i < mats.size() ; i++)
    { 
        const G4Material* mt = mats[i]  ; 
        const G4String& mtname = mt->GetName() ;
        const char* mtn = mtname.c_str(); 
        NPFold* matfold = MakePropertyFold(mt) ; 
        fold->add_subfold( mtn, matfold ); 
    }  
    return fold ; 
}


NPFold* U4Material::MakePropertyFold(const G4Material* mat )
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

G4MaterialPropertyVector* U4Material::MakeProperty( double value )
{
    NP* a = MakePropertyArray(value); 
    return MakeProperty(a); 
}

NP* U4Material::MakePropertyArray( double value )
{
    sdomain sdom ;  
    std::vector<double> dom(sdom.energy_eV, sdom.energy_eV+sdom.length) ; 
    for(unsigned i=0 ; i < dom.size() ; i++)  dom[i] *= eV ; 
    std::reverse( dom.begin(), dom.end() ); 

    NP* a = NP::Make<double>(sdom.length, 2); 
    double* vv = a->values<double>(); 
    for(unsigned i=0 ; i < sdom.length ; i++)
    {
        vv[i*2+0] = dom[i] ; 
        vv[i*2+1] = value  ; 
    }
    return a ; 
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

TODO: Use of IDPath is not acceptable in new workflow 


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
U4Material::LoadBnd from $CFBase/CSGFoundry/SSim/bnd.npy so must have already saved it
----------------------------------------------------------------------------------------

SSim::Load loads from $CFBase/CSGFoundry/SSim where "$CFBase" is an 
internal envvar that yield the result of SOpticksResource::CFBase()
The OPTICKS_KEY derived CSG_GGeo directory is returned unless 
the CFBASE envvar is defined. 

HMM: if the material exists already then need to change its 
properties, not scrub the pre-existing material. 
Thats needed for scintillators as the standard bnd properties
do not include all the needed scint props. 

Load the material properties from the SSim::get_bnd array using SBnd::getPropertyGroup 
for each material. 

**/

void U4Material::LoadBnd(const char* ssimdir)
{
    SSim* sim = SSim::Load(ssimdir);  
    const SBnd* sb = sim->get_sbnd(); 
    if(sb == nullptr) 
    {
        LOG(fatal) << "failed to load bnd.npy from ssimdir " << ssimdir ; 
        return ;  
    }

    std::vector<std::string> mnames ; 
    sb->getMaterialNames(mnames );

    std::vector<std::string> pnames ; 
    SBnd::GetMaterialPropNames(pnames); 

    sdomain sdom ;  
    std::vector<double> dom(sdom.energy_eV, sdom.energy_eV+sdom.length) ; 
    for(unsigned i=0 ; i < dom.size() ; i++)  dom[i] *= eV ; 


    for(unsigned m=0 ; m < mnames.size() ; m++) 
    {
        const char* mname = mnames[m].c_str() ; 
        bool warning = false ; 
        G4Material* prior = G4Material::GetMaterial(mname, warning);  
        if(prior != nullptr) std::cout << "prior material " << mname << std::endl ; 

        G4Material* mat = prior ? prior : MakeWater(mname) ; 
        G4MaterialPropertiesTable* mpt = mat->GetMaterialPropertiesTable()  ; 
        if(mpt == nullptr) 
        {
            mpt = new G4MaterialPropertiesTable ; 
            mat->SetMaterialPropertiesTable(mpt);
        }

        for(unsigned p=0 ; p < pnames.size() ; p++) 
        {
            const char* pname = pnames[p].c_str() ; 
            std::vector<double> val ; 
            sb->getProperty(val, mname, pname); 
            assert( val.size() == dom.size() ); 

            bool reverse = true ; 
            AddProperty(mpt, pname, dom, val, reverse ); 
        }
    }
}


int U4Material::GetIndex(const std::vector<G4String>& nn, const char* key ) // static
{
    G4String k(key);
    typedef std::vector<G4String> VS ;
    typedef VS::const_iterator   VSI ;
    VSI b = nn.begin() ;
    VSI e = nn.end() ;
    VSI p = std::find(b, e, k );
    return p == e ? -1 : std::distance(b, p) ;
}

int U4Material::GetPropertyIndex( const G4MaterialPropertiesTable* mpt, const char* key ) // static
{
    const std::vector<G4String> names = mpt->GetMaterialPropertyNames() ;
    return GetIndex(names, key); 
}

G4MaterialPropertyVector* U4Material::AddProperty( G4MaterialPropertiesTable* mpt, const char* key, const std::vector<double>& dom, const std::vector<double>& val , bool reverse )
{
     std::vector<double> ldom(dom); 
     std::vector<double> lval(val); 
     assert( ldom.size() == lval.size() ); 
     if(reverse)
     {
         std::reverse(ldom.begin(), ldom.end()); 
         std::reverse(lval.begin(), lval.end()); 
     }
     unsigned numval = ldom.size(); 
     double* ldom_v = ldom.data() ; 
     double* lval_v = lval.data() ; 

#if G4VERSION_NUMBER < 1100
    G4MaterialPropertyVector* mpv = mpt->AddProperty(key, ldom_v, lval_v, numval );
#else
    int keyIdx = GetPropertyIndex(mpt, key);
    G4bool createNewKey = keyIdx == -1  ;
    G4MaterialPropertyVector* mpv = mpt->AddProperty(key, ldom_v, lval_v, numval, createNewKey );
#endif
    return mpv ; 
}


/**
U4Material::KludgeRemoveRockRINDEX
-----------------------------------

Removing the Rock RINDEX property is a kludge that makes photons immediately get absorbed 
on reaching the Rock. However this kludge is problematic to random aligned running as
the random consumption does not then follow the normal pattern. 

In order for the tail random consumption to be amenable to aligning with Opticks
instead add perfect absorber surfaces to have a cleaner termination instead of the "kill"
that happens with NoRINDEX.   

For example U4VolumeMaker::RaindropRockAirWater
adds a surface using U4Surface::MakePerfectAbsorberSurface in 

**/

void U4Material::KludgeRemoveRockRINDEX() // static
{
    RemoveProperty( "RINDEX", G4Material::GetMaterial("Rock") ); 
}


