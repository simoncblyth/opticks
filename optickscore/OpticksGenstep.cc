#include <iostream>
#include <iomanip>

#include "NGLM.hpp"
#include "GLMFormat.hpp"
#include "GLMPrint.hpp"
#include "NPY.hpp"
#include "NSlice.hpp"

#include "OpticksPhoton.h"
#include "OpticksFlags.hh"
#include "OpticksGenstep.hh"

#include "PLOG.hh"

OpticksGenstep::OpticksGenstep(NPY<float>* gs) 
    :  
    m_gensteps(gs),
    m_gensteps_content_version(gs ? gs->getArrayContentVersion() : -1 ), 
    m_num_gensteps(gs ? gs->getNumItems() : 0),
    m_num_photons(gs ? gs->getUSum(0,3) : 0),
    m_avg_photons_per_genstep( m_num_gensteps > 0 ? float(m_num_photons)/float(m_num_gensteps) : 0 )
{
    init();
}

void OpticksGenstep::init()
{
    assert( m_gensteps->hasShape(-1,6,4) );
}

unsigned OpticksGenstep::getNumGensteps() const { return m_num_gensteps ; }
unsigned OpticksGenstep::getNumPhotons() const { return m_num_photons ; }
float OpticksGenstep::getAvgPhotonsPerGenstep() const { return m_avg_photons_per_genstep ; }


unsigned OpticksGenstep::getGencode(unsigned idx) const 
{
    int gs00 = m_gensteps->getInt(idx,0u,0u) ;

    int gencode = -1 ; 
    if( m_gensteps_content_version == 0 )  // old style unversioned gensteps , this is fallback when no metadata 
    {
        gencode = gs00 < 0 ? CERENKOV : SCINTILLATION ;  
    }
    else if( m_gensteps_content_version == 1042 )
    {
        gencode = gs00 ; 
    }
    else
    { 
        LOG(fatal) << " unexpected gensteps_content_version " << m_gensteps_content_version ; 
        assert(0); 
    }

    bool expected = gencode == CERENKOV || gencode == SCINTILLATION  ; 

    if(!expected)
         LOG(fatal) << "unexpected gencode " 
                    << " gencode " << gencode
                    << " flag " << OpticksFlags::Flag(gencode) 
                    ;

    assert(expected) ; 
    return gencode ; 
}



NPY<float>* OpticksGenstep::getGensteps() const { return m_gensteps ; }

glm::ivec4 OpticksGenstep::getHdr(unsigned i) const 
{
    glm::ivec4 hdr = m_gensteps->getQuadI(i,0);
    return hdr ; 
}
glm::vec4 OpticksGenstep::getPositionTime(unsigned i) const 
{
    glm::vec4 post = m_gensteps->getQuad(i,1);
    return post ; 
}
glm::vec4 OpticksGenstep::getDeltaPositionStepLength(unsigned i) const 
{
    glm::vec4 dpsl = m_gensteps->getQuad(i,2);
    return dpsl ; 
}


glm::vec4 OpticksGenstep::getQ3(unsigned i) const  {  return m_gensteps->getQuad(i,3); }
glm::vec4 OpticksGenstep::getQ4(unsigned i) const  {  return m_gensteps->getQuad(i,4); }
glm::vec4 OpticksGenstep::getQ5(unsigned i) const  {  return m_gensteps->getQuad(i,5); }

glm::ivec4 OpticksGenstep::getI3(unsigned i) const  {  return m_gensteps->getQuadI(i,3); }
glm::ivec4 OpticksGenstep::getI4(unsigned i) const  {  return m_gensteps->getQuadI(i,4); }
glm::ivec4 OpticksGenstep::getI5(unsigned i) const  {  return m_gensteps->getQuadI(i,5); }


std::string OpticksGenstep::desc() const 
{
    std::stringstream ss ;
    ss << "OpticksGenstep " 
       << ( m_gensteps ? m_gensteps->getShapeString() : "-" ) 
       << " gensteps_content_version " << m_gensteps_content_version
       << " num_gensteps " << m_num_gensteps
       << " num_photons " << m_num_photons
       << " avg_photons_per_genstep " << m_avg_photons_per_genstep
       ; 
    return ss.str();
}

std::string OpticksGenstep::desc(unsigned i) const 
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


void OpticksGenstep::Dump(NPY<float>* gs_, unsigned modulo, unsigned margin, const char* msg) 
{
    LOG(info) << msg 
              << " modulo " << modulo
              << " margin " << margin
              << " gs_ " << ( gs_ ? "Y" : "NULL" ) 
              ;
 
    if(!gs_) return ; 
    OpticksGenstep gs(gs_) ;
    gs.dump(modulo, margin); 
}


void OpticksGenstep::dump(unsigned modulo, unsigned margin, const char* msg) const
{
    NSlice slice(0, getNumGensteps()) ;

    LOG(info) << msg 
              << " slice " << slice.description()
              << " modulo " << modulo
              << " margin " << margin 
              << std::endl 
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


