#include "CGDMLDetector.hh"

// npy-
#include "NLog.hpp"
#include "GLMFormat.hpp"

// ggeo-
#include "GCache.hh"

// g4-
#include "G4LogicalVolume.hh"
#include "G4ThreeVector.hh"
#include "G4PVPlacement.hh"
#include "G4UImanager.hh"

#include "globals.hh"
#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"


// painfully this is not standard in G4
#include "G4GDMLParser.hh"


void CGDMLDetector::init()
{
}

G4VPhysicalVolume* CGDMLDetector::Construct()
{
    G4VPhysicalVolume* top = NULL ;

    const char* gdmlpath = m_cache->getGDMLPath();
    LOG(info) << "CGDMLDetector::Construct " << gdmlpath ; 


    bool validate = false ; 
    G4GDMLParser parser;
    parser.Read(gdmlpath, validate);

    top = parser.GetWorldVolume();

/*
   As have seen previously XML ids must not include certain characters, causing validation failures
    env/geant4/geometry/gdml/g4pygdml.py

G4GDML: VALIDATION ERROR! Datatype error: Type:InvalidDatatypeValueException, Message:Value 'cylinder+ChildForsource-assy0xc2d5788_pos' is not valid NCName . at line: 1393
G4GDML: VALIDATION ERROR! Datatype error: Type:InvalidDatatypeValueException, Message:Value 'cylinder+ChildForsource-assy0xc2d5788_rot' is not valid NCName . at line: 1394
*/
  
    return top ; 
}
 
