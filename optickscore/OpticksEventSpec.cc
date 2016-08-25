#include <cstring>

#include "NLoad.hpp"
#include "NPY.hpp"

#include "OpticksEventSpec.hh"

#include "PLOG.hh"

OpticksEventSpec::OpticksEventSpec(OpticksEventSpec* spec) 
    :
    m_typ(spec->getTyp()),
    m_tag(spec->getTag()),
    m_det(spec->getDet()),
    m_cat(spec->getCat())
{
}

OpticksEventSpec::OpticksEventSpec(const char* typ, const char* tag, const char* det, const char* cat) 
          :
          m_typ(strdup(typ)),
          m_tag(strdup(tag)),
          m_det(strdup(det)),
          m_cat(cat ? strdup(cat) : NULL)
{
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






