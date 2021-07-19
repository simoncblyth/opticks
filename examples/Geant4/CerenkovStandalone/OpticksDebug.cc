#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;

#include "NP.hh"

#include "G4Material.hh"
#include "G4MaterialPropertiesTable.hh"
#include "G4MaterialPropertyVector.hh"
#include "G4SystemOfUnits.hh"

#include "OpticksDebug.hh"


NP* OpticksDebug::LoadArray(const char* kdpath) // static
{
    const char* keydir = getenv("OPTICKS_KEYDIR") ; 
    assert( keydir ); 
    std::stringstream ss ; 
    ss << keydir << "/" << kdpath ;  
    std::string s = ss.str(); 
    const char* path = s.c_str(); 
    std::cout << "OpticksDebug::LoadArray " << path << std::endl ; 
    NP* a = NP::Load(path); 
    return a ; 
}

G4MaterialPropertyVector* OpticksDebug::MakeProperty(const NP* a)  // static
{
    unsigned nv = a->num_values() ; 
    std::cout << "a " << a->desc() << " num_values " << nv << std::endl ; 
    const double* vv = a->values<double>() ; 

    assert( nv %  2 == 0 ); 
    unsigned entries = nv/2 ; 
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

G4Material* OpticksDebug::MakeMaterial(G4MaterialPropertyVector* rindex)  // static
{
    // its Water, but that makes no difference for Cerenkov 
    // the only thing that matters us the rindex property
    G4double a, z, density;
    G4int nelements;
    G4Element* O = new G4Element("Oxygen"  , "O", z=8 , a=16.00*CLHEP::g/CLHEP::mole);
    G4Element* H = new G4Element("Hydrogen", "H", z=1 , a=1.01*CLHEP::g/CLHEP::mole);
    G4Material* mat = new G4Material("Water", density= 1.0*CLHEP::g/CLHEP::cm3, nelements=2);
    mat->AddElement(H, 2);
    mat->AddElement(O, 1);

    rindex->SetSpline(false);
    //rindex->SetSpline(true);

    G4MaterialPropertiesTable* mpt = new G4MaterialPropertiesTable();
    mpt->AddProperty("RINDEX", rindex );   
    mat->SetMaterialPropertiesTable(mpt) ;
    return mat ; 
}



OpticksDebug::OpticksDebug(unsigned itemsize_, const char* name_)
    :
    itemsize(itemsize_), 
    name(strdup(name_))
{
}

void OpticksDebug::append( double x, const char* name )
{
    values.push_back(x);  
    if( names.size() <= itemsize ) names.push_back(name); 
} 
void OpticksDebug::append( unsigned x, unsigned y, const char* name )
{
    assert( sizeof(unsigned)*2 == sizeof(double) ); 
    DUU duu ; 
    duu.uu.x = x ; 
    duu.uu.y = y ;            // union trickery to place two unsigned into the slot of a double  
    append(duu.d, name); 
}

void OpticksDebug::write(const char* dir, const char* reldir, unsigned nj, unsigned nk )
{
    bool expected_size = values.size() % itemsize == 0  ; 
    unsigned ni = values.size() / itemsize  ; 
    assert( nj*nk == itemsize ); 

    if(!expected_size)
    {
       std::cout 
           << " UNEXPECTED SIZE "
           << " values.size " << values.size()
           << " itemsize " << itemsize 
           << " ni " << ni
           << std::endl 
           ;
    }
    assert( expected_size ); 

    std::string stem = name ; 
    std::string npy = stem + ".npy" ; 
    std::string txt = stem + ".txt" ; 

    std::cout 
        << "OpticksDebug::write"
        << " ni " << ni 
        << " dir " << dir
        << " reldir " << reldir
        << std::endl
        ; 

    if( ni > 0 )
    {
        NP::Write(     dir, reldir, npy.c_str(), values.data(), ni, nj, nk ); 
        NP::WriteNames(dir, reldir, txt.c_str(), names, itemsize ); 
    }
}


bool OpticksDebug::ExistsPath(const char* base_, const char* reldir_, const char* name_ )
{
    fs::path fpath(base_);
    if(reldir_) fpath /= reldir_ ;
    if(name_) fpath /= name_ ;
    bool x = fs::exists(fpath); 
    std::cout << "OpticksDebug::ExistsPath " << ( x ? "Y" : "N" ) << " " << fpath.string().c_str() << std::endl ; 
    return x ; 
}

int OpticksDebug::getenvint(const char* envkey, int fallback)
{
    char* val = getenv(envkey);
    int ival = val ? std::atoi(val) : fallback ;
    return ival ; 
}


std::string OpticksDebug::prepare_path(const char* dir_, const char* reldir_, const char* name )
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
OpticksDebug::ListDir
-----------------------

From BDir::dirlist, collect names of files within directory *path* with names ending with *ext*.

**/

void OpticksDebug::ListDir(std::vector<std::string>& names,  const char* path, const char* ext) // static
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




