// cfg4--;op --cgdmldetector
#include "CGDMLDetector.hh"

// boost-
#include <boost/algorithm/string.hpp>

// npy-
#include "NLog.hpp"
#include "GLMFormat.hpp"

// ggeo-
#include "GCache.hh"
#include "GMaterial.hh"

// cfg4-
#include "CPropLib.hh"
#include "CTraverser.hh"

// g4-
#include "G4LogicalVolume.hh"
#include "G4ThreeVector.hh"
#include "G4PVPlacement.hh"
#include "G4UImanager.hh"

#include "globals.hh"
#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"
#include "G4Material.hh"

// painfully this is not standard in G4
#include "G4GDMLParser.hh"


void CGDMLDetector::init()
{
    m_lib = new CPropLib(m_cache);
}

G4VPhysicalVolume* CGDMLDetector::Construct()
{
    const char* gdmlpath = m_cache->getGDMLPath();
    LOG(info) << "CGDMLDetector::Construct " << gdmlpath ; 

    bool validate = false ; 

    G4GDMLParser parser;
    parser.Read(gdmlpath, validate);

    m_top = parser.GetWorldVolume();

    fixMaterials(m_top);

    return m_top ; 
}

void CGDMLDetector::fixMaterials(G4VPhysicalVolume* top)
{
    m_traverser = new CTraverser(top); 
    m_traverser->Traverse();
    m_traverser->Summary();

    unsigned int nmat = m_traverser->getNumMaterials();
    unsigned int nmat_without_mpt = m_traverser->getNumMaterialsWithoutMPT();

    if(nmat > 0 && nmat_without_mpt == nmat )
    {
        LOG(warning) << "CGDMLDetector::fixMaterials" 
                     << " ALL G4 MATERIALS LACK MPT "
                     << " FIXING USING G4DAE MATERIALS " 
                     ;

         addMPT();
    } 
    else if(nmat > 0 && nmat_without_mpt == 0) 
    {
        assert(0); 
    } 
    else if(nmat > 0 && nmat_without_mpt > 0 ) 
    {
        assert(0); 
    } 
    else 
    {
        assert(0); 
    }

}



void CGDMLDetector::addMPT()
{
    // GDML exported by geant4 that comes with nuwa lack material properties 
    // so use the properties from the G4DAE export 

    unsigned int ng4mat = m_traverser->getNumMaterialsWithoutMPT() ;
    for(unsigned int i=0 ; i < ng4mat ; i++)
    {
        G4Material* g4mat = m_traverser->getMaterialWithoutMPT(i) ;
        const char* name = g4mat->GetName() ;

        std::vector<std::string> elem;
        boost::split(elem,name,boost::is_any_of("/"));
        assert(elem.size() == 4 && "expecting material names like /dd/Materials/GdDopedLS " );
        const char* shortname = elem[3].c_str();

        const GMaterial* ggmat = m_lib->getMaterial(shortname);          
        assert(ggmat && strcmp(ggmat->getShortName(), shortname)==0 && "failed to find corresponding G4DAE material") ;

        LOG(info) << "CGDMLDetector::addMPT" 
                  << " g4mat " << std::setw(45) << name
                  << " shortname " << std::setw(25) << shortname
                   ;

        G4MaterialPropertiesTable* mpt = m_lib->makeMaterialPropertiesTable(ggmat);
        g4mat->SetMaterialPropertiesTable(mpt);
         
    }
}










 
