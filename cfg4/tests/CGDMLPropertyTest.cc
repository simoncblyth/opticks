#include <vector>
#include "OPTICKS_LOG.hh"
#include "OKConf.hh"
#include "SPath.hh"

#include "G4GDMLParser.hh"
#include "G4VPhysicalVolume.hh"
#include "G4Material.hh"
#include "G4LogicalBorderSurface.hh"
#include "G4LogicalSkinSurface.hh"
#include "G4OpticalSurface.hh"
#include "G4MaterialPropertiesTable.hh"
#include "G4String.hh"

#include "X4LogicalBorderSurfaceTable.hh"

#include "CGDML.hh"

/**
CGDMLPropertyTest
===================

See CDumpTest which uses CDump.cc/CDump.hh that 
makes the dump functions started here easier to share.

::

   opticksaux-;cp $(opticksaux-dx1) /tmp/v1.gdml

   CGDMLPropertyTest /tmp/v1.gdml


See notes/issues/g4-1062-geocache-create-reflectivity-assert.rst 

Observe some difference between Geant4 1042 and 1062 that prevents 
the geometry conversion for 1062. Looks like efficiency values
all coming out as zero in 1062 that prevents identification as a sensor.

**/

void dump_MPT(const char* name, const G4MaterialPropertiesTable* mpt, unsigned edgeitems)
{
    std::cout << name << " " ; 
    if(mpt == NULL) std::cout << " NULL mpt " << std::endl ; 
    if(mpt == NULL) return ; 

    std::vector<G4String> pns = mpt->GetMaterialPropertyNames() ; 
    std::cout  << pns.size() << std::endl ; 

    typedef G4MaterialPropertyVector MPV ; 
    for(unsigned i=0 ; i < pns.size() ; i++)
    {
        const std::string& pname = pns[i] ; 
        G4int pidx = mpt->GetPropertyIndex(pname);
        MPV* pvec = const_cast<G4MaterialPropertiesTable*>(mpt)->GetProperty(pidx);
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

void dump_G4LogicalBorderSurfaceTable(unsigned edgeitems)
{
    unsigned nlbs = G4LogicalBorderSurface::GetNumberOfBorderSurfaces() ;
    const G4LogicalBorderSurfaceTable* tab_ = G4LogicalBorderSurface::GetSurfaceTable() ; 
    const std::vector<G4LogicalBorderSurface*>* tab = X4LogicalBorderSurfaceTable::PrepareVector(tab_); 

    LOG(info) << " nlbs " << nlbs << " tab.size " << tab->size() ; 

    for(size_t i=0 ; i < tab->size() ; i++)
    {   
        G4LogicalBorderSurface* src = (*tab)[i] ; 

        G4OpticalSurface* opsurf = dynamic_cast<G4OpticalSurface*>(src->GetSurfaceProperty());
        assert(opsurf); 

        const G4String& name = src->GetName(); 
        const G4MaterialPropertiesTable* mpt = opsurf->GetMaterialPropertiesTable(); 
        dump_MPT(name.c_str(), mpt, edgeitems); 
    }   
}


void dump_G4LogicalSkinSurface(unsigned edgeitems)
{
    unsigned nlss = G4LogicalSkinSurface::GetNumberOfSkinSurfaces() ;
    const G4LogicalSkinSurfaceTable* tab = G4LogicalSkinSurface::GetSurfaceTable();
    LOG(info) << " nlss " << nlss << " tab.size " << tab->size() ; 

    for(size_t i=0 ; i < tab->size() ; i++)
    {   
        G4LogicalSkinSurface* src = (*tab)[i] ; 

        G4OpticalSurface* opsurf = dynamic_cast<G4OpticalSurface*>(src->GetSurfaceProperty());
        assert(opsurf); 

        const G4String& name = src->GetName(); 
        const G4MaterialPropertiesTable* mpt = opsurf->GetMaterialPropertiesTable(); 
        dump_MPT(name.c_str(), mpt, edgeitems); 
    }   
}

void dump_G4MaterialTable(unsigned edgeitems)
{
    unsigned nmat = G4Material::GetNumberOfMaterials();
    const G4MaterialTable* tab = G4Material::GetMaterialTable();
    LOG(info) << " nmat " << nmat << " tab.size " << tab->size() ; 

    for(size_t i=0 ; i < tab->size() ; i++)
    {   
        G4Material* src = (*tab)[i] ; 
        const G4String& name = src->GetName(); 
        const G4MaterialPropertiesTable* mpt = src->GetMaterialPropertiesTable(); 
        dump_MPT(name.c_str(), mpt, edgeitems ); 
    }
}


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    LOG(info) << "OKConf::Geant4VersionInteger() : " << OKConf::Geant4VersionInteger()  ;

    const char* path_ = argc > 1 ? argv[1] : "$TMP/v1.gdml" ; 
    const char* path = SPath::Resolve(path_, 0); 

    if(!path) LOG(error) << " expecting path to GDML " ; 
    if(!path) return 0 ; 
    bool readable = SPath::IsReadable(path) ; 
    if(!readable)  LOG(error) << " path is not readable " << path ; 
    if(!readable) return 0 ; 

    LOG(info) << " parsing " << path ; 
    G4VPhysicalVolume* world = CGDML::Parse(path); 
    assert( world ); 

    unsigned edgeitems = 5; 

    dump_G4LogicalBorderSurfaceTable(edgeitems); 
    dump_G4LogicalSkinSurface(edgeitems);
    dump_G4MaterialTable(edgeitems) ; 

    return 0 ; 
}

// om-;TEST=CGDMLPropertyTest;om-t 

