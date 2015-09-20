#include "TorchStepNPY.hpp"
#include "NPY.hpp"

#include <vector>
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal


void TorchStepNPY::parseConfig()
{
    std::string cfgline = m_config ; 
    std::vector<std::string> elem ;
    boost::split(elem, cfgline, boost::is_any_of(","));

    for(unsigned int i=0 ; i < elem.size() ; i++)
    {
        std::vector<std::string> kv ; 
        boost::split(kv, elem[i], boost::is_any_of(":"));
        assert(kv.size() == 2);
        if(strcmp(kv[0].c_str(), "target")==0) m_target = boost::lexical_cast<int>(kv[1]) ; 
    }


    LOG(info) << "TorchStepNPY::parseConfig "
              << " target " << m_target 
              ;
}



NPY<float>* TorchStepNPY::makeNPY()
{
    m_npy = NPY<float>::make(1, 6, 4);
    m_npy->zero();

    // see cu/torchstep.h

    m_npy->setQuadI(0, 0, m_ctrl );
    m_npy->setQuad( 0, 1, m_post );
    m_npy->setQuad( 0, 2, m_dirw );
    m_npy->setQuad( 0, 3, m_polw );

    return m_npy ; 
}

