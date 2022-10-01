#include <iostream>
#include <iomanip>
#include <cstring>
#include <sstream>
#include <xercesc/util/PlatformUtils.hpp>

#include "GDXMLRead.hh"
#include "GDXMLWrite.hh"
#include "GDXML.hh"

#include "SStr.hh"
#include "SLOG.hh"

const plog::Severity GDXML::LEVEL = SLOG::EnvLevel("GDXML", "DEBUG" ); 

/**
GDXML::Fix
-------------------

1. reads *srcpath* using xercesc 
2. examines the xml at xercesc level to find issues 
3. fixes the issues  
4. writes the fixed xml to *dstpath*

Formerly only wrote when issues to fix, but 
that complicates usage. 

Instead of doing that its better to have a raw intermediate .gdml
so the user who is not paying attention can be unaware of the fixup. 
But file organization is left to the user.  

**/

void GDXML::Fix(const char* dstpath, const char* srcpath)  // static
{
    xercesc::XMLPlatformUtils::Initialize();  // HMM: might clash with Geant4 ? 

    bool same = strcmp(dstpath, srcpath) == 0 ; 
    assert( same == false ); 
    GDXML gd(srcpath);  
    gd.write(dstpath);  
}


GDXML::GDXML(const char* srcpath_)
    :
    srcpath(strdup(srcpath_)),
    kludge_truncated_matrix(true), 
    reader(new GDXMLRead(srcpath, kludge_truncated_matrix)), 
    doc(const_cast<xercesc::DOMDocument*>(reader->doc)), 
    defineElement(reader->the_defineElement), 
    num_duplicated_matrixElement(reader->checkDuplicatedMatrix()),
    num_pruned_matrixElement(reader->pruneDuplicatedMatrix()),
    num_truncated_matrixElement(reader->truncated_matrixElement.size()),
    num_constants(reader->constants.size()), 
    writer(new GDXMLWrite(doc)),
    issues(false) 
{
    if(num_constants > 0 ) replaceAllConstantWithMatrix(); 
    issues = (num_truncated_matrixElement > 0 || num_constants > 0 ) ;
}

std::string GDXML::desc() const 
{
    std::stringstream ss ; 
    ss << "GDXML::desc" << std::endl 
       << " srcpath " << srcpath << std::endl 
       << " num_duplicated_matrixElement " << num_duplicated_matrixElement << std::endl 
       << " num_pruned_matrixElement " << num_pruned_matrixElement << std::endl  
       << " num_truncated_matrixElement " << num_truncated_matrixElement << std::endl 
       << " num_constants " << num_constants << std::endl 
       << " issues " << ( issues ? "YES" : "NO" ) << std::endl 
       ;
    std::string s = ss.str(); 
    return s ; 
}

void GDXML::write(const char* dstpath)
{
    const char* txtpath = SStr::ReplaceEnd(dstpath, ".gdml", "_gdxml_report.txt" ); 
    LOG(LEVEL) << "writing .gdml dstpath " << dstpath ; 
    LOG(LEVEL) << "writing .txt report txtpath " <<  txtpath ;
    writer->write(dstpath); 
    std::string rep = desc(); 
    LOG(LEVEL) << " rep " << std::endl << rep ; 
    SStr::Save(txtpath, rep.c_str() ); 
}

GDXML::~GDXML()
{
}

void GDXML::replaceAllConstantWithMatrix()
{
    assert( defineElement );  
    for(unsigned i=0 ; i < num_constants ; i++)
    {
        const Constant& c = reader->constants[i] ; 
        LOG(LEVEL)
            << " c.name " << std::setw(20) << c.name 
            << " c.value " << std::setw(10) << c.value 
            ; 

        double nm_lo = 80. ; 
        double nm_hi = 800. ; 
        xercesc::DOMElement* matrixElement = writer->ConstantToMatrixElement(c.name.c_str(), c.value, nm_lo, nm_hi ); 
        defineElement->appendChild(matrixElement);
    }
}


