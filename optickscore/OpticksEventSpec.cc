#include <cstring>
#include <string>
#include <sstream>

#include "BStr.hh"
#include "BOpticksEvent.hh"

#include "NLoad.hpp"
#include "NPY.hpp"

#include "OpticksEventSpec.hh"

#include "PLOG.hh"

const char* OpticksEventSpec::G4_ = "G4" ; 
const char* OpticksEventSpec::OK_ = "OK" ; 
const char* OpticksEventSpec::NO_ = "NO" ; 


OpticksEventSpec::OpticksEventSpec(OpticksEventSpec* spec) 
    :
    m_typ(spec->getTyp()),
    m_tag(spec->getTag()),
    m_det(spec->getDet()),
    m_cat(spec->getCat()),
    m_udet(spec->getUDet()),
    m_dir(NULL),
    m_fold(NULL),
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
    m_udet(cat && strlen(cat) > 0 ? strdup(cat) : strdup(det)),
    m_dir(NULL),
    m_fold(NULL),
    m_itag(BStr::atoi(m_tag, 0))
{
    init();
}


OpticksEventSpec* OpticksEventSpec::clone(unsigned tagoffset)
{
    int itag = getITag();
    assert(itag != 0);
    int ntag = itag > 0 ? itag + tagoffset : itag - tagoffset ; 
    const char* tag = BStr::itoa( ntag );
    return new OpticksEventSpec( getTyp(), tag, getDet(), getCat() );
}

void OpticksEventSpec::init()
{
    const char* udet = getUDet();    
    std::string tagdir = NLoad::directory(udet, m_typ, m_tag ) ; 
    std::string typdir = NLoad::directory(udet, m_typ, NULL ) ; 
    m_dir = strdup(tagdir.c_str());
    m_fold = strdup(typdir.c_str());
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

const char* OpticksEventSpec::getEngine()
{
    const char* engine = NO_ ; 
    if(     isOK())  engine = OK_ ; 
    else if(isG4())  engine = G4_ ; 
    return engine ; 
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
    return m_udet ; 
}
const char* OpticksEventSpec::getDir()
{
    return m_dir ; 
}
const char* OpticksEventSpec::getFold()
{
    return m_fold ; 
}


std::string OpticksEventSpec::brief()
{
    std::stringstream ss ; 
    ss 
       << " typ " << m_typ
       << " tag " << m_tag
       << " itag " << getITag()
       << " det " << m_det
       << " cat " << m_cat
       << " dir " << m_dir
       << " eng " << getEngine()
       ;

    return ss.str();
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
