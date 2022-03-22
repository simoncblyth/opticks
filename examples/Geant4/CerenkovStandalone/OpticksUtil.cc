
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
    std::vector<double> d, v ; 
    a->psplit<double>(d,v);   // split array into domain and values 
    assert(d.size() == v.size() && d.size() > 1 ); 

    G4MaterialPropertyVector* mpv = new G4MaterialPropertyVector(d.data(), v.data(), d.size() ); 
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
The names are sorted using default std::sort lexical ordering. 

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

/**
OpticksUtil::LoadConcat
-------------------------

If *concat_path* ends with ".npy" simply loads it into seq array
otherwise *concat_path* is assumed to be a directory containing multiple ".npy"
to be concatenated.  The names of the paths in the directory are obtained using 
OpticksUtil::ListDir 

**/

NP* OpticksUtil::LoadConcat(const char* concat_path)
{
    NP* seq = nullptr ; 
    if(concat_path && ExistsPath(concat_path))
    {   
        if(strlen(concat_path) > 4 && strcmp(concat_path+strlen(concat_path)-4, ".npy") == 0)
        {
            seq = NP::Load(concat_path);  
        }   
        else
        {
            std::vector<std::string> names ; 
            ListDir(names, concat_path, ".npy");   
            std::cout 
                << "OpticksUtil::LoadRandom" 
                << " directory " << concat_path 
                << " contains names.size " << names.size() 
                << " .npy" 
                << std::endl
                ;  
            seq = NP::Concatenate(concat_path, names); 
        }
    }
    else
    {
        std::cout 
            << "OpticksUtil::LoadRandom"
            << " non-existing concat_path " << ( concat_path ? concat_path : "-" ) 
            << std::endl
            ;   
    }
    return seq ; 
}


