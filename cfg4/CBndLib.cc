#include "OpticksHub.hh"

#include "GBndLib.hh"
#include "GMaterialLib.hh"
#include "GSurLib.hh"

#include "CBndLib.hh"

CBndLib::CBndLib(OpticksHub* hub)
    :
    m_hub(hub),
    m_bndlib(m_hub->getBndLib()),
    m_matlib(m_hub->getMaterialLib()),
    m_surlib(m_hub->getSurLib())
{
}


unsigned CBndLib::addBoundary(const char* spec)
{
    return m_bndlib->addBoundary(spec); 
}

GMaterial* CBndLib::getOuterMaterial(unsigned boundary)
{
    unsigned omat_ = m_bndlib->getOuterMaterial(boundary);
    GMaterial* omat = m_matlib->getMaterial(omat_);
    return omat ; 
}

GMaterial* CBndLib::getInnerMaterial(unsigned boundary)
{
    unsigned imat_ = m_bndlib->getInnerMaterial(boundary);
    GMaterial* imat = m_matlib->getMaterial(imat_);
    return imat ; 
}

GSur* CBndLib::getOuterSurface(unsigned boundary)
{
    unsigned osur_ = m_bndlib->getOuterSurface(boundary);
    return m_surlib->getSur(osur_);
}
GSur* CBndLib::getInnerSurface(unsigned boundary)
{
    unsigned isur_ = m_bndlib->getInnerSurface(boundary);
    return m_surlib->getSur(isur_);
}




