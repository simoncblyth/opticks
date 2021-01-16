#include <string>
#include <iostream>
#include <iomanip>

#include "G4Version.hh"
#include "G4Material.hh"
#include "G4LogicalBorderSurface.hh"
#include "G4LogicalSkinSurface.hh"
#include "G4OpticalSurface.hh"
#include "G4MaterialPropertiesTable.hh"
#include "G4String.hh"

#include "X4LogicalBorderSurfaceTable.hh"
#include "X4Dump.hh"

const unsigned X4Dump::EDGEITEMS = 5 ; 



void X4Dump::G4(const char* cfg) // static
{
    std::cout << cfg << std::endl ;
    X4Dump::G4Version_() ; 
    std::istringstream f(cfg);
    std::string s;
    char delim = ',' ; 
    while (getline(f, s, delim))
    {   
        std::cout << s << std::endl ; 
        if(s.compare("bs") == 0) X4Dump::G4LogicalBorderSurfaceTable_();  ;
        if(s.compare("sk") == 0) X4Dump::G4LogicalSkinSurface_() ;
        if(s.compare("mt") == 0) X4Dump::G4MaterialTable_() ; 
    }   
    X4Dump::G4Version_() ; 
}

void X4Dump::G4Version_()  // static 
{
    std::cout << "G4VERSION_NUMBER " << G4VERSION_NUMBER << std::endl ; 
    std::cout << "G4VERSION_TAG    " << G4VERSION_TAG << std::endl ; 
    std::cout << "G4Version        " << G4Version << std::endl ; 
    std::cout << "G4Date           " << G4Date << std::endl ; 
}

void X4Dump::G4MaterialPropertiesTable_(const char* name, const G4MaterialPropertiesTable* mpt) // static
{
    unsigned edgeitems = EDGEITEMS ; 
    std::cout << name << " " ; 
    if(mpt == NULL) std::cout << " NULL mpt " << std::endl ; 
    if(mpt == NULL) return ; 

    std::vector<G4String> pns = mpt->GetMaterialPropertyNames() ; 
    std::cout  << pns.size() << std::endl ; 

    typedef G4MaterialPropertyVector MPV ; 
    for(unsigned i=0 ; i < pns.size() ; i++)
    {
        const std::string& pname = pns[i] ; 
        bool warning = false ; 
        G4int pidx = mpt->GetPropertyIndex(pname, warning);
        MPV* pvec = const_cast<G4MaterialPropertiesTable*>(mpt)->GetProperty(pidx, warning );
        size_t plen = pvec ? pvec->GetVectorLength() : 0 ;
        if(pvec != NULL)
        {
            std::cout 
                << " i " << std::setw(5) << i
                << " pidx " << std::setw(5) << pidx
                << " " << std::setw(23) << pname 
                << " pvec " << std::setw(8) << pvec 
                << " plen " << std::setw(3) << plen 
                << " " 
                ; 

            double mn = std::numeric_limits<double>::max();   
            double mx = std::numeric_limits<double>::lowest(); 

            for (size_t j=0; j<plen; j++)
            {   
                double value = (*pvec)[j] ;
                if(value > mx) mx = value ; 
                if(value < mn) mn = value ; 

                if( j < edgeitems || j > plen - edgeitems ) std::cout << value << " " ; 
                else if( j == edgeitems ) std::cout << "... " ; 
            }
            std::cout << " mn " << mn << " mx " << mx << std::endl ; 
        }
    }
}

void X4Dump::G4LogicalBorderSurfaceTable_() // static 
{
    unsigned nlbs = G4LogicalBorderSurface::GetNumberOfBorderSurfaces() ;
    const G4LogicalBorderSurfaceTable* tab = G4LogicalBorderSurface::GetSurfaceTable() ; 
    const std::vector<G4LogicalBorderSurface*>* vec = X4LogicalBorderSurfaceTable::PrepareVector(tab); 

    std::cout << " nlbs " << nlbs << " vec.size " << vec->size() << std::endl  ; 

    for(size_t i=0 ; i < vec->size() ; i++)
    {   
        G4LogicalBorderSurface* src = (*vec)[i] ; 

        G4OpticalSurface* opsurf = dynamic_cast<G4OpticalSurface*>(src->GetSurfaceProperty());
        assert(opsurf); 

        const G4String& name = src->GetName(); 
        const G4MaterialPropertiesTable* mpt = opsurf->GetMaterialPropertiesTable(); 
        X4Dump::G4MaterialPropertiesTable_(name.c_str(), mpt) ; 
    }   
}


void X4Dump::G4LogicalSkinSurface_()  // static
{
    unsigned nlss = G4LogicalSkinSurface::GetNumberOfSkinSurfaces() ;
    const G4LogicalSkinSurfaceTable* tab = G4LogicalSkinSurface::GetSurfaceTable();
    std::cout << " nlss " << nlss << " tab.size " << tab->size() << std::endl ; 

    for(size_t i=0 ; i < tab->size() ; i++)
    {   
        G4LogicalSkinSurface* src = (*tab)[i] ; 

        G4OpticalSurface* opsurf = dynamic_cast<G4OpticalSurface*>(src->GetSurfaceProperty());
        assert(opsurf); 

        const G4String& name = src->GetName(); 
        const G4MaterialPropertiesTable* mpt = opsurf->GetMaterialPropertiesTable(); 
        X4Dump::G4MaterialPropertiesTable_(name.c_str(), mpt); 
    }   
}

void X4Dump::G4MaterialTable_() // static
{
    unsigned nmat = G4Material::GetNumberOfMaterials();
    const G4MaterialTable* tab = G4Material::GetMaterialTable();
    std::cout << " nmat " << nmat << " tab.size " << tab->size() << std::endl ; 

    for(size_t i=0 ; i < tab->size() ; i++)
    {   
        G4Material* src = (*tab)[i] ; 
        const G4String& name = src->GetName(); 
        const G4MaterialPropertiesTable* mpt = src->GetMaterialPropertiesTable(); 
        X4Dump::G4MaterialPropertiesTable_(name.c_str(), mpt ); 
    }
}


