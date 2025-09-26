
#include "U4Polycone.h"
#include "G4SystemOfUnits.hh"
#include "U4Mesh.h"

#include "NPFold.h"
#include "s_csg.h"


struct U4Polycone_test
{
    static int Flange();
    static int Mug();
    static int Main();
};

/**
U4Polycone_test::Flange
------------------------

                     :
        +------------:-------------+
        |            :             |
        +-----+      :      +------+
              |      :      |
              +------:------+
                     :
**/
inline int U4Polycone_test::Flange()
{
    const char* name = "Flange" ;

    G4double phiStart = 0.00*deg ;
    G4double phiTotal = 360.00*deg ;

    G4int numRZ = 4 ;
    G4double ri[] = {0. ,  0.,  0. , 0.    } ;
    //G4double ro[] = {50. , 50., 100., 100. } ;
    G4double ro[] = {264.050, 264.050, 264.050, 132.025 } ;
    G4double z[] = { -183.225,      0., 100.  , 200.050 } ;

    G4Polycone* pc = new G4Polycone(name, phiStart, phiTotal, numRZ, z, ri, ro );
    //G4cout << *pc << std::endl ;

    U4Mesh::Save(pc, "$FOLD", name);


    NPFold* fold = nullptr ;

    s_csg* csg = new s_csg ;
    assert(csg);

    int lvid = 0 ;
    int depth = 0 ;
    int level = 1 ;
    sn* root = U4Polycone::Convert( pc, lvid, depth, level );
    std::cout << root->render() ;

    std::cout << csg->brief() << std::endl ;
    fold = csg->serialize();
    fold->save("$FOLD", name, "_csg");

    return 0 ;
}

/**
U4Polycone_test::Mug
------------------------


             2
            +-+                                                                   +-+  22000
            | |                                                                   | |
            | |                                                                   | |
            | |                                                                   | |
            | |                                                                   | |
            | |                                                                   | |
            | |                                                                   | |
            | |                                                                   | |
            | |                                                                   | |
            | |                                                                   | |
            | |                                                                   | |
            | |                                                                   | |
            | |                                                                   | |
            | |                                                                   | |
            | |                                                                   | |
            | |                                                                   | |
            | |                                                                   | |
            | |                                                                   | |
            + +-------------------------------+-----------------------------------+ +  -21650
            +---------------------------------+-------------------------------------+  -21652


**/


inline int U4Polycone_test::Mug()
{
    const char* name = "Mug" ;

    G4double phiStart = 0.00*deg ;
    G4double phiTotal = 360.00*deg ;
    G4int numRZ = 4 ;
    G4double ri[] = { 0.       ,       0.,  21650.0 , 21650.0 } ;
    G4double ro[] = { 21652.0  ,  21652.0,  21652.0,  21652.0 } ;
    G4double z[] =  { -21652.0 , -21650.0, -21650.0 , 22000.0 } ;

    G4Polycone* pc = new G4Polycone(name, phiStart, phiTotal, numRZ, z, ri, ro );
    //G4cout << *pc << std::endl ;

    U4Mesh::Save(pc, "$FOLD", name);

    int lvid = 0 ;
    int depth = 0 ;
    int level = 1 ;
    sn* root = U4Polycone::Convert( pc, lvid, depth, level );
    std::cout << root->render() ;


    return 0 ;
}


inline int U4Polycone_test::Main()
{
    const char* TEST = ssys::getenvvar("TEST","Mug");
    bool ALL = strcmp(TEST, "ALL") == 0 ;
    int rc = 0 ;
    if(ALL||0==strcmp(TEST,"Flange")) rc += Flange();
    if(ALL||0==strcmp(TEST,"Mug"))    rc += Mug();
    return rc ;
}

int main(){ return U4Polycone_test::Main() ;  }

