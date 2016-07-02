//
// with DO_LOG not defined (for a clean environment)
//
// without ITERATOR_DEBUGGING disabled
// there is an abort (and nasty dialog box pops up)
// on attempting to pass std::string to the G4 (RelWithDebInfo) library 
//
// with ITERATOR_DEBUGGING disabled 
// this completes without error
//
// 
// with DO_LOG defined a segfault occurs at the end of main
// with or without CFG4_LOG__ in play 
// running in VS with "opticks-vs" from PowerShell  
// reveals an access violation in G4SolidStore::Clean
//
//  this is bizarre logging is messing up the G4SolidStore   
//


#define _HAS_ITERATOR_DEBUGGING 0
#define DO_LOG 1


#include <string>
#include <iostream>
#include <cassert>

#include "CFG4_BODY.hh"
#include "G4Box.hh"



#ifdef DO_LOG
#include "PLOG.hh"
//#include "CFG4_LOG.hh"
#endif


void test_G4Box_0()
{
    std::cerr << "test_G4Box" << std::endl ; 

    G4Box* box = new G4Box("TestBox", 1.,1.,1.); 

    std::cerr << "test_G4Box (after ctor)" << std::endl ; 

    std::cerr << "test_G4Box"
               << " name " << box->GetName() 
               << std::endl ;

   // delete box ; 

    std::cerr << "test_G4Box DONE" << std::endl ; 
}


void test_G4Box_1()
{
    std::cerr << "test_G4Box" << std::endl ; 


    std::string name = "G4Box_1" ;

    G4Box* box = new G4Box(name, 1.,1.,1.); 

    std::cerr << "test_G4Box (after ctor)" << std::endl ; 

    std::cerr << "test_G4Box"
               << " name " << box->GetName() 
               << std::endl ;


    //delete box ; 


    std::cerr << "test_G4Box DONE" << std::endl ; 
}





int main(int argc, char** argv)
{
#ifdef DO_LOG
    PLOG_(argc, argv);
  //  CFG4_LOG__ ;
    LOG(info) << argv[0] ;
#endif


    test_G4Box_0(); 
    test_G4Box_1(); 


    std::cerr << "test_G4Box back in main " << std::endl ; 

#ifdef DO_LOG
    LOG(info) << argv[0] << " DONE " ;
#endif



    return 0 ;
}

