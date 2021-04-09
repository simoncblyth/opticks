//name=GDMLKludgeFixMatrixTruncation ; xercesc- ; gcc $name.cc GDMLRead.cc GDMLWrite.cc  -I$(xercesc-prefix)/include -L$(xercesc-prefix)/lib -lxerces-c -std=c++11 -lstdc++ -o /tmp/$name && /tmp/$name $HOME/origin2.gdml
/**

   g4-
   g4-cls G4GDMLRead
   g4-cls G4GDMLReadDefine

**/
#include <iostream>
#include <iomanip>
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


    xercesc::DOMDocument* doc = const_cast<xercesc::DOMDocument*>(reader.doc); 

    GDMLWrite writer(doc); 


    xercesc::DOMElement*  defineElement = reader.the_defineElement ; 
    assert( defineElement );  

    unsigned num_constants = reader.constants.size() ; 
    for(unsigned i=0 ; i < num_constants ; i++)
    {
        const Constant& c = reader.constants[i] ; 
        std::cout 
            << " c.name " << std::setw(20) << c.name 
            << " c.value " << std::setw(10) << c.value 
            << std::endl
            ; 

        double nm_lo = 80. ; 
        double nm_hi = 800. ; 
        xercesc::DOMElement* matrixElement = writer.ConstantToMatrixElement(c.name.c_str(), c.value, nm_lo, nm_hi ); 
        defineElement->appendChild(matrixElement);
    }

    writer.write(outpath); 

    return 0 ; 
}


