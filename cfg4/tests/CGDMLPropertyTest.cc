#include <vector>
#include "OPTICKS_LOG.hh"
#include "OKConf.hh"

#include "G4GDMLParser.hh"
#include "G4VPhysicalVolume.hh"
#include "G4Material.hh"
#include "G4LogicalBorderSurface.hh"
#include "G4OpticalSurface.hh"
#include "G4MaterialPropertiesTable.hh"
#include "G4String.hh"

#include "CGDML.hh"

/**

::

   opticksaux-;cp $(opticksaux-dx1) /tmp/v1.gdml

   CGDMLPropertyTest /tmp/v1.gdml


See notes/issues/g4-1062-geocache-create-reflectivity-assert.rst 

Observe some difference between Geant4 1042 and 1062 that prevents 
the geometry conversion for 1062. Looks like efficiency values
all coming out as zero in 1062 that prevents identification as a sensor.

**/

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv);

    LOG(info) << "OKConf::Geant4VersionInteger() : " << OKConf::Geant4VersionInteger()  ;

    const char* path = argc > 1 ? argv[1] : "/tmp/v1.gdml" ; 

    if(!path) LOG(error) << " expecting path to GDML " ; 
    if(!path) return 0 ; 

    LOG(info) << " parsing " << path ; 
    G4VPhysicalVolume* world = CGDML::Parse(path); 
    assert( world ); 

    unsigned nmat = G4Material::GetNumberOfMaterials();
    LOG(info) << " nmat " << nmat ; 

    unsigned nlbs = G4LogicalBorderSurface::GetNumberOfBorderSurfaces() ;
    LOG(info) << " nlbs " << nlbs ; 

    // X4LogicalBorderSurfaceTable
    const G4LogicalBorderSurfaceTable*  m_src = G4LogicalBorderSurface::GetSurfaceTable() ; 
   
    for(size_t i=0 ; i < m_src->size() ; i++)
    {   
        G4LogicalBorderSurface* src = (*m_src)[i] ; 


        // X4LogicalBorderSurface
        G4OpticalSurface* opsurf = dynamic_cast<G4OpticalSurface*>(src->GetSurfaceProperty());
        assert(opsurf); 

        G4MaterialPropertiesTable* mpt = opsurf->GetMaterialPropertiesTable() ; 
        assert(mpt);  

        std::vector<G4String> pns = mpt->GetMaterialPropertyNames() ; 


        LOG(info) << src->GetName() << " "  << pns.size(); 

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
                    << " pname" << std::setw(30) << pname 
                    << " pvec " << std::setw(8) << pvec 
                    << " plen " << std::setw(8) << plen 
                    << std::endl ; 

                for (size_t j=0; j<plen; j++)
                {       
                    double value = (*pvec)[j] ;
                    std::cout << value << " " ; 
                }
                std::cout << std::endl ; 

            }


        }



    }   


    return 0 ; 
}

// om-;TEST=CGDMLPropertyTest;om-t 

