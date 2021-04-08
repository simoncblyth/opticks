//name=GDMLKludgeFixMatrixTruncation ; xercesc- ; gcc $name.cc GDMLRead.cc GDMLWrite.cc  -I$(xercesc-prefix)/include -L$(xercesc-prefix)/lib -lxerces-c -std=c++11 -lstdc++ -o /tmp/$name && /tmp/$name $HOME/origin2.gdml
/**

   g4-
   g4-cls G4GDMLRead
   g4-cls G4GDMLReadDefine

**/
#include <iostream>
#include "GDMLRead.hh"
#include "GDMLWrite.hh"

int main(int argc, char** argv)
{
    const char* inpath = argc > 1 ? argv[1] : nullptr ; 
    const char* outpath = argc > 2 ? argv[2] : "/tmp/out.gdml" ; 
    if(!inpath) return 0 ; 

    xercesc::XMLPlatformUtils::Initialize();

    bool kludge_truncated_matrix = true ; 
    GDMLRead reader(inpath, kludge_truncated_matrix);  

    unsigned num_truncated_matrixElement = reader.truncated_matrixElement.size(); 
    std::cout << "num_truncated_matrixElement " << num_truncated_matrixElement << std::endl ; 

    GDMLWrite writer(reader.doc); 
    writer.write(outpath); 

    return 0 ; 
}


