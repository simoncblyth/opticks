#include "NGLM.hpp"
#include "NPY.hpp"
#include "GLMFormat.hpp"
#include "OpticksDomain.hh"

#include "PLOG.hh"




OpticksDomain::OpticksDomain()
          :
          m_fdom(NULL),
          m_idom(NULL)
{
    init();
}

void OpticksDomain::init()
{
}


void OpticksDomain::setFDomain(NPY<float>* fdom)
{
    m_fdom = fdom ; 
}
void OpticksDomain::setIDomain(NPY<int>* idom)
{
    m_idom = idom ; 
}

NPY<float>* OpticksDomain::getFDomain()
{
    return m_fdom ; 
}
NPY<int>* OpticksDomain::getIDomain()
{
    return m_idom ; 
}



void OpticksDomain::setSpaceDomain(const glm::vec4& space_domain)
{
    m_space_domain = space_domain ; 
}
void OpticksDomain::setTimeDomain(const glm::vec4& time_domain)
{
    m_time_domain = time_domain  ; 
}
void OpticksDomain::setWavelengthDomain(const glm::vec4& wavelength_domain)
{
    m_wavelength_domain = wavelength_domain  ; 
}


const glm::vec4& OpticksDomain::getSpaceDomain()
{
    return m_space_domain ; 
}
const glm::vec4& OpticksDomain::getTimeDomain()
{
    return m_time_domain ;
}
const glm::vec4& OpticksDomain::getWavelengthDomain()
{ 
    return m_wavelength_domain ; 
}





void OpticksDomain::updateBuffer()
{
    NPY<float>* fdom = getFDomain();
    if(fdom)
    {
        fdom->setQuad(m_space_domain     , 0);
        fdom->setQuad(m_time_domain      , 1);
        fdom->setQuad(m_wavelength_domain, 2);
    }
    else
    {
        LOG(warning) << "OpticksDomain::updateBuffer fdom NULL " ;
    }

    NPY<int>* idom = getIDomain();
    if(idom)
        idom->setQuad(m_settings, 0 );
    else
        LOG(warning) << "OpticksDomain::updateBuffer idom NULL " ;
    
}


void OpticksDomain::importBuffer()
{
    NPY<float>* fdom = getFDomain();
    assert(fdom);
    m_space_domain = fdom->getQuad(0);
    m_time_domain = fdom->getQuad(1);
    m_wavelength_domain = fdom->getQuad(2);

    if(m_space_domain.w <= 0.)
    {
        LOG(fatal) << "OpticksDomain::importBuffer BAD FDOMAIN" ; 
        dump("OpticksDomain::importBuffer");
        assert(0);
    }

    NPY<int>* idom = getIDomain();
    assert(idom);

    m_settings = idom->getQuad(0); 
    unsigned int maxrec = m_settings.w ; 

    if(maxrec != 10)
        LOG(fatal) << "OpticksDomain::importBuffer" 
                   << " from idom settings m_maxrec BUT EXPECT 10 " << maxrec 
                        ;
    //assert(maxrec == 10);
}







unsigned OpticksDomain::getMaxRng() const 
{
    return m_settings.y ; 
}
void OpticksDomain::setMaxRng(unsigned maxrng)
{
    m_settings.y = maxrng ; 
}

unsigned int OpticksDomain::getMaxBounce() const 
{
    return m_settings.z ; 
}
void OpticksDomain::setMaxBounce(unsigned int maxbounce)
{
    m_settings.z = maxbounce ; 
}

unsigned int OpticksDomain::getMaxRec() const 
{
    return m_settings.w ; 
}
void OpticksDomain::setMaxRec(unsigned int maxrec)
{
    m_settings.w = maxrec ; 
}










void OpticksDomain::dump(const char* msg)
{
    LOG(info) << msg 
              << "\n space_domain      " << gformat(m_space_domain)
              << "\n time_domain       " << gformat(m_time_domain)
              << "\n wavelength_domain " << gformat(m_wavelength_domain)
              ;
}


