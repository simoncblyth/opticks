#include <iostream>
#include <iomanip>

#include "NGLM.hpp"
#include "GLMFormat.hpp"
#include "GLMPrint.hpp"
#include "NPY.hpp"
#include "NSlice.hpp"
#include "NGS.hpp"

#include "PLOG.hh"

NGS::NGS(NPY<float>* gensteps) 
    :  
    m_gensteps(gensteps)
{
    init();
}

void NGS::init()
{
    assert( m_gensteps->hasShape(-1,6,4) );
}

unsigned NGS::getNumGensteps() const 
{
    return m_gensteps ? m_gensteps->getNumItems() : 0  ; 
}

NPY<float>* NGS::getGensteps() const 
{
    return m_gensteps ; 
}

glm::ivec4 NGS::getHdr(unsigned i) const 
{
    glm::ivec4 hdr = m_gensteps->getQuadI(i,0);
    return hdr ; 
}
glm::vec4 NGS::getPositionTime(unsigned i) const 
{
    glm::vec4 post = m_gensteps->getQuad(i,1);
    return post ; 
}

glm::vec4 NGS::getDeltaPositionStepLength(unsigned i) const 
{
    glm::vec4 dpsl = m_gensteps->getQuad(i,2);
    return dpsl ; 
}

glm::vec4 NGS::getQ3(unsigned i) const 
{
    glm::vec4 q = m_gensteps->getQuad(i,3);
    return q ; 
}
glm::vec4 NGS::getQ4(unsigned i) const 
{
    glm::vec4 q = m_gensteps->getQuad(i,4);
    return q ; 
}
glm::vec4 NGS::getQ5(unsigned i) const 
{
    glm::vec4 q = m_gensteps->getQuad(i,5);
    return q ; 
}


std::string NGS::desc() const 
{
    std::stringstream ss ;
    ss << "NGS " << ( m_gensteps ? m_gensteps->getShapeString() : "-" ) ; 
    return ss.str();
}

std::string NGS::desc(unsigned i) const 
{
    glm::ivec4 hdr = getHdr(i);
    glm::vec4 post = getPositionTime(i);
    glm::vec4 dpsl = getDeltaPositionStepLength(i);

    std::stringstream ss ;
    ss 
        << " i " << std::setw(7) << i 
        << " hdr " << std::setw(20) << gpresent(hdr) 
        << " post " << std::setw(20) << gpresent(post) 
        << " dpsl " << std::setw(20) << gpresent(dpsl) 
        ;

    return ss.str();
}


void NGS::Dump(NPY<float>* gs_, unsigned modulo, unsigned margin, const char* msg) 
{
    LOG(info) << msg 
              << " modulo " << modulo
              << " margin " << margin
              << " gs_ " << ( gs_ ? "Y" : "NULL" ) 
              ;
 
    if(!gs_) return ; 
    NGS gs(gs_) ;
    gs.dump(modulo, margin); 
}


void NGS::dump(unsigned modulo, unsigned margin, const char* msg) const
{
    NSlice slice(0, getNumGensteps()) ;

    LOG(info) << msg 
              << " slice " << slice.description()
              << " modulo " << modulo
              << " margin " << margin 
              << " desc " << desc() 
              ; 

    for(unsigned i=slice.low ; i < slice.high ; i += slice.step )
    {
        if(slice.isMargin(i, margin) || i % modulo == 0)
        {
            std::cout << desc(i) << std::endl ; 
        }
    }
}


