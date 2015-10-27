#include "GBndLib.hh"
#include "GPropertyMap.hh"

#include "GMaterialLib.hh"
#include "GSurfaceLib.hh"

#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal


guint4 GBndLib::getOrCreate(
               GPropertyMap<float>* imat_,  
               GPropertyMap<float>* omat_,  
               GPropertyMap<float>* isur_,  
               GPropertyMap<float>* osur_)
{

    unsigned int imat = m_mlib->getIndex(imat_->getShortName()) ;
    unsigned int omat = m_mlib->getIndex(omat_->getShortName()) ;
    unsigned int isur = m_slib->getIndex(isur_ ? isur_->getShortName() : NULL) ;
    unsigned int osur = m_slib->getIndex(osur_ ? osur_->getShortName() : NULL) ;

    guint4 bnd = guint4(imat, omat, isur, osur);

    //m_bnd.insert(bnd);   need to add cmp function to guint4 

    //std::set<guint4>::iterator it;
    //it = m_bnd.find(bnd);
    // hmm get index in the bnd ? and use that 

    //  std::distance( m_bnd.begin() , it )
    //  hmm but this will change whilst growing 
    // need unique sequence useing insert order ? 


    return bnd ; 
}


void GBndLib::saveToCache()
{
    LOG(info) << "GBndLib::saveToCache placeholder" ;
}

 
