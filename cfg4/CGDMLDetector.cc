// cfg4--;op --cgdmldetector

// boost-
#include "CFG4_BODY.hh"
#include <boost/algorithm/string.hpp>


#include "BFile.hh"


// npy-
#include "GLMFormat.hpp"

// ggeo-
#include "GMaterial.hh"
#include "GSurfaceLib.hh"

// okc-
#include "Opticks.hh"
#include "OpticksResource.hh"

#include "OpticksHub.hh"   // okg-

// cfg4-
#include "CMaterialLib.hh"
#include "CTraverser.hh"

// g4-
#include "G4LogicalVolume.hh"
#include "G4Material.hh"
#include "G4GDMLParser.hh"

#include "CGDMLDetector.hh"

#include "PLOG.hh"


CGDMLDetector::CGDMLDetector(OpticksHub* hub, OpticksQuery* query)
  : 
  CDetector(hub, query)
{
    init();
}

CGDMLDetector::~CGDMLDetector()
{
}


void CGDMLDetector::init()
{
    const char* path = m_ok->getGDMLPath();
    bool exists = BFile::ExistsFile(path);
    std::string npath = BFile::FormPath(path);

    LOG(info) << "CGDMLDetector::init" 
              << " path " << path
              << " npath " << npath
              ; 


    if(!exists)
    {
         LOG(error)
              << "CGDMLDetector::init" 
              << " PATH DOES NOT EXIST "
              << " path " << path
              << " npath " << npath
              ; 

         setValid(false);  

         return ; 
     }



    bool validate = false ; 

    G4String gpath = npath ; 
    LOG(trace) << "parse " << gpath ; 


    G4GDMLParser parser;
    parser.Read(gpath, validate);

    setTop(parser.GetWorldVolume());   // invokes *CDetector::traverse*

    addMPT();
}


void CGDMLDetector::saveBuffers()
{
    // split off to allow setting idpath override whilst testing
    CDetector::saveBuffers("CGDMLDetector", 0);    
}


void CGDMLDetector::addMPT()
{
    // GDML exported by geant4 that comes with nuwa lack material properties 
    // so use the properties from the G4DAE export 

    unsigned int nmat = m_traverser->getNumMaterials();
    unsigned int nmat_without_mpt = m_traverser->getNumMaterialsWithoutMPT();

    if(nmat > 0 && nmat_without_mpt == nmat )
    {
        LOG(warning) << "CGDMLDetector::addMPT" 
                     << " ALL G4 MATERIALS LACK MPT "
                     << " FIXING USING G4DAE MATERIALS " 
                     ;
    } 
    else
    {
        LOG(fatal) << "CGDMLDetector::addMPT UNEXPECTED"
                   << " nmat " << nmat 
                   << " nmat_without_mpt " << nmat_without_mpt
                   ;
        assert(0);
    }
 

    unsigned int ng4mat = m_traverser->getNumMaterialsWithoutMPT() ;
    for(unsigned int i=0 ; i < ng4mat ; i++)
    {
        G4Material* g4mat = m_traverser->getMaterialWithoutMPT(i) ;
        const char* name = g4mat->GetName() ;

        std::vector<std::string> elem;
        boost::split(elem,name,boost::is_any_of("/"));
        assert(elem.size() == 4 && "expecting material names like /dd/Materials/GdDopedLS " );
        const char* shortname = elem[3].c_str();

        const GMaterial* ggmat = m_mlib->getMaterial(shortname);          
        assert(ggmat && strcmp(ggmat->getShortName(), shortname)==0 && "failed to find corresponding G4DAE material") ;

        LOG(debug) << "CGDMLDetector::addMPT" 
                  << " g4mat " << std::setw(45) << name
                  << " shortname " << std::setw(25) << shortname
                   ;

        G4MaterialPropertiesTable* mpt = m_mlib->makeMaterialPropertiesTable(ggmat);
        g4mat->SetMaterialPropertiesTable(mpt);
        //m_mlib->dumpMaterial(g4mat, "CGDMLDetector::addMPT");        
         
    }

    LOG(info) << "CGDMLDetector::addMPT added MPT to " <<  ng4mat << " g4 materials " ; 

}



 
