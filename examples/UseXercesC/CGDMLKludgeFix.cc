#include <iostream>
#include <iomanip>
#include <cstring>
#include <sstream>

#include "CGDMLRead.hh"
#include "CGDMLWrite.hh"
#include "CGDMLKludgeFix.hh"


/**
ReplaceEnd
------------------

String s is required to have ending q.
New string n is returned with the ending q replaced with r.

**/

const char* ReplaceEnd( const char* s, const char* q, const char* r  )
{
    int pos = strlen(s) - strlen(q) ;
    assert( pos > 0 && strncmp(s + pos, q, strlen(q)) == 0 );

    std::stringstream ss ; 
    for(int i=0 ; i < pos ; i++) ss << *(s+i) ;   
    ss << r ; 

    std::string n = ss.str(); 
    return strdup(n.c_str());
}



CGDMLKludgeFix::CGDMLKludgeFix(const char* srcpath_)
    :
    srcpath(strdup(srcpath_)),
    dstpath(ReplaceEnd(srcpath, ".gdml", "_CGDMLKludgeFix.gdml")),
    kludge_truncated_matrix(true), 
    reader(new CGDMLRead(srcpath, kludge_truncated_matrix)), 
    doc(const_cast<xercesc::DOMDocument*>(reader->doc)), 
    defineElement(reader->the_defineElement), 
    num_truncated_matrixElement(reader->truncated_matrixElement.size()),
    num_constants(reader->constants.size()), 
    writer(new CGDMLWrite(doc)) 
{
    std::cout 
        << "num_truncated_matrixElement " << num_truncated_matrixElement 
        << "num_constants " << num_constants
        << std::endl
        ; 
    
    if(num_constants > 0 )
    {
        replaceAllConstantWithMatrix();
    }

    if(num_truncated_matrixElement > 0 || num_constants > 0 )
    {
       std::cout << "writing dstpath " << dstpath << std::endl ; 
       writer->write(dstpath); 
    }
    else
    {
       std::cout << "found no problems that needed kludging : NOT writing dstpath " << dstpath << std::endl ; 
    }
}


CGDMLKludgeFix::~CGDMLKludgeFix()
{
}


void CGDMLKludgeFix::replaceAllConstantWithMatrix()
{
    assert( defineElement );  
    for(unsigned i=0 ; i < num_constants ; i++)
    {
        const Constant& c = reader->constants[i] ; 
        std::cout 
            << "CGDMLKludgeFix::AddMatrixElements" 
            << " c.name " << std::setw(20) << c.name 
            << " c.value " << std::setw(10) << c.value 
            << std::endl
            ; 

        double nm_lo = 80. ; 
        double nm_hi = 800. ; 
        xercesc::DOMElement* matrixElement = writer->ConstantToMatrixElement(c.name.c_str(), c.value, nm_lo, nm_hi ); 
        defineElement->appendChild(matrixElement);
    }
}


