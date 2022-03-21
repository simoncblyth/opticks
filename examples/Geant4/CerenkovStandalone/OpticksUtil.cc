
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;

#include "OpticksUtil.hh"

#include "NP.hh"

#include "G4Material.hh"
#include "G4MaterialPropertiesTable.hh"
#include "G4MaterialPropertyVector.hh"
#include "G4SystemOfUnits.hh"


NP* OpticksUtil::LoadArray(const char* kdpath) // static
{
    const char* keydir = getenv("OPTICKS_KEYDIR") ; 
    assert( keydir ); 
    std::stringstream ss ; 
    ss << keydir << "/" << kdpath ;  
    std::string s = ss.str(); 
    const char* path = s.c_str(); 
    std::cout << "OpticksUtil::LoadArray " << path << std::endl ; 
    NP* a = NP::Load(path); 
    return a ; 
}


G4MaterialPropertyVector* OpticksUtil::MakeProperty(const NP* a)  // static
{
    unsigned nv = a->num_values() ; 
    std::cout << "a " << a->desc() << " num_values " << nv << std::endl ; 
    const double* vv = a->cvalues<double>() ; 

    assert( nv %  2 == 0 ); 
    unsigned entries = nv/2 ;

    // this has to be double for G4 
    std::vector<double> e(entries, 0.); 
    std::vector<double> v(entries, 0.); 

    for(unsigned i=0 ; i < entries ; i++)
    {
        e[i] = vv[2*i+0] ; 
        v[i] = vv[2*i+1] ; 
        std::cout 
            << " e[i]/eV " << std::fixed << std::setw(10) << std::setprecision(4) << e[i]/eV
            << " v[i] " << std::fixed << std::setw(10) << std::setprecision(4) << v[i] 
            << std::endl 
            ;     
    }
    G4MaterialPropertyVector* mpv = new G4MaterialPropertyVector(e.data(), v.data(), entries ); 
    return mpv ; 
}




G4Material* OpticksUtil::MakeMaterial(const G4MaterialPropertyVector* rindex, const char* name)  // static
{
    // its Water, but that makes no difference for Cerenkov 
    // the only thing that matters us the rindex property
    G4double a, z, density;
    G4int nelements;
    G4Element* O = new G4Element("Oxygen"  , "O", z=8 , a=16.00*CLHEP::g/CLHEP::mole);
    G4Element* H = new G4Element("Hydrogen", "H", z=1 , a=1.01*CLHEP::g/CLHEP::mole);
    G4Material* mat = new G4Material(name, density= 1.0*CLHEP::g/CLHEP::cm3, nelements=2);
    mat->AddElement(H, 2);
    mat->AddElement(O, 1);

    G4MaterialPropertiesTable* mpt = new G4MaterialPropertiesTable();

    G4MaterialPropertyVector* rindex_ = const_cast<G4MaterialPropertyVector*>(rindex) ;  // HUH: why not const ?
    mpt->AddProperty("RINDEX", rindex_ );   
    mat->SetMaterialPropertiesTable(mpt) ;
    return mat ; 
}


bool OpticksUtil::ExistsPath(const char* base_, const char* reldir_, const char* name_ )
{
    fs::path fpath(base_);
    if(reldir_) fpath /= reldir_ ;
    if(name_) fpath /= name_ ;
    bool x = fs::exists(fpath); 
    std::cout << "OpticksUtil::ExistsPath " << ( x ? "Y" : "N" ) << " " << fpath.string().c_str() << std::endl ; 
    return x ; 
}

int OpticksUtil::getenvint(const char* envkey, int fallback)
{
    char* val = getenv(envkey);
    int ival = val ? std::atoi(val) : fallback ;
    return ival ; 
}


std::string OpticksUtil::prepare_path(const char* dir_, const char* reldir_, const char* name )
{   
    fs::path fdir(dir_);
    if(reldir_) fdir /= reldir_ ;

    if(!fs::exists(fdir))
    {   
        if (fs::create_directories(fdir))
        {   
            std::cout << "created directory " << fdir.string().c_str() << std::endl  ;
        }   
    }   

    fs::path fpath(fdir); 
    fpath /= name ;   

    return fpath.string();
}


/**
OpticksUtil::ListDir
-----------------------

From BDir::dirlist, collect names of files within directory *path* with names ending with *ext*.

**/

void OpticksUtil::ListDir(std::vector<std::string>& names,  const char* path, const char* ext) // static
{
    fs::path dir(path);
    if(!( fs::exists(dir) && fs::is_directory(dir))) return ; 

    fs::directory_iterator it(dir) ;
    fs::directory_iterator end ;
   
    for(; it != end ; ++it)
    {   
        std::string fname = it->path().filename().string() ;
        const char* fnam = fname.c_str();

        if(strlen(fnam) > strlen(ext) && strcmp(fnam + strlen(fnam) - strlen(ext), ext)==0)
        {   
            names.push_back(fnam);
        }   
    }   

    std::sort( names.begin(), names.end() ); 
}

NP* OpticksUtil::LoadRandom(const char* random_path)
{
    NP* seq = nullptr ; 
    if(random_path && ExistsPath(random_path))
    {   
        if(strlen(random_path) > 4 && strcmp(random_path+strlen(random_path)-4, ".npy") == 0)
        {
            seq = NP::Load(random_path);  
        }   
        else
        {
            std::vector<std::string> names ; 
            ListDir(names, random_path, ".npy");   
            std::cout 
                << "OpticksUtil::LoadRandom" 
                << " directory " << random_path 
                << " contains names.size " << names.size() 
                << " .npy" 
                << std::endl
                ;  
            seq = NP::Concatenate(random_path, names); 
        }
    }
    else
    {
        std::cout 
            << "OpticksUtil::LoadRandom"
            << " non-existing random_path " << ( random_path ? random_path : "-" ) 
            << std::endl
            ;   
    }
    return seq ; 
}


