#include <cstring>
#include <string>

#include "BStr.hh"
#include "BOpticksEvent.hh"

#include "NLoad.hpp"
#include "NPY.hpp"

#include "OpticksEventSpec.hh"

#include "PLOG.hh"

OpticksEventSpec::OpticksEventSpec(OpticksEventSpec* spec) 
    :
    m_typ(spec->getTyp()),
    m_tag(spec->getTag()),
    m_det(spec->getDet()),
    m_cat(spec->getCat()),
    m_dir(NULL),
    m_itag(spec->getITag())
{
    init();
}

OpticksEventSpec::OpticksEventSpec(const char* typ, const char* tag, const char* det, const char* cat) 
    :
    m_typ(strdup(typ)),
    m_tag(strdup(tag)),
    m_det(strdup(det)),
    m_cat(cat ? strdup(cat) : NULL),
    m_dir(NULL),
    m_itag(BStr::atoi(m_tag, 0))
{
    init();
}




void OpticksEventSpec::init()
{
    const char* udet = getUDet();    
    std::string tagdir = NLoad::directory(udet, m_typ, m_tag ) ; 
    m_dir = strdup(tagdir.c_str());
}

int OpticksEventSpec::getITag()
{
    return m_itag ; 
}
bool OpticksEventSpec::isG4()
{
    return m_itag < 0 ;     
}
bool OpticksEventSpec::isOK()
{
    return m_itag > 0 ;     
}

const char* OpticksEventSpec::getTyp()
{
    return m_typ ; 
}
const char* OpticksEventSpec::getTag()
{
    return m_tag ; 
}
const char* OpticksEventSpec::getDet()
{
    return m_det ; 
}
const char* OpticksEventSpec::getCat()
{
    return m_cat ; 
}
const char* OpticksEventSpec::getUDet()
{
    return m_cat && strlen(m_cat) > 0 ? m_cat : m_det ; 
}
const char* OpticksEventSpec::getDir()
{
    return m_dir ; 
}




void OpticksEventSpec::Summary(const char* msg)
{
    LOG(info) << msg 
              << " typ " << m_typ
              << " tag " << m_tag
              << " itag " << getITag()
              << " det " << m_det
              << " cat " << m_cat
              << " dir " << m_dir
              ;
}
