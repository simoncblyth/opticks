#include <cstring>
#include <string>

#include "NLoad.hpp"
#include "NPY.hpp"

#include "OpticksEventSpec.hh"
#include "BOpticksEvent.hh"

#include "PLOG.hh"

OpticksEventSpec::OpticksEventSpec(OpticksEventSpec* spec) 
    :
    m_typ(spec->getTyp()),
    m_tag(spec->getTag()),
    m_det(spec->getDet()),
    m_cat(spec->getCat()),
    m_dir(NULL)
{
    init();
}

OpticksEventSpec::OpticksEventSpec(const char* typ, const char* tag, const char* det, const char* cat) 
    :
    m_typ(strdup(typ)),
    m_tag(strdup(tag)),
    m_det(strdup(det)),
    m_cat(cat ? strdup(cat) : NULL),
    m_dir(NULL)
{
    init();
}


void OpticksEventSpec::init()
{
    const char* udet = getUDet();    
    std::string tagdir = NLoad::directory(udet, m_typ, m_tag ) ; 
    m_dir = strdup(tagdir.c_str());
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
              << " det " << m_det
              << " cat " << m_cat
              << " dir " << m_dir
              ;
}
