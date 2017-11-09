#include "BStr.hh"
#include "PLOG.hh"

#include "GMaterialLib.hh"
#include "GSurfaceLib.hh"
#include "GBnd.hh"

GBnd::GBnd(const char* spec_, bool flip_, GMaterialLib* mlib_, GSurfaceLib* slib_, bool dbgbnd_ ) 
    : 
    UNSET(mlib_->getUNSET()),
    spec(strdup(spec_)),
    mlib(mlib_),
    slib(slib_),
    dbgbnd(dbgbnd_)
{
   init(flip_);
   check();
}

void GBnd::init(bool flip_)
{
    BStr::split(elem, spec, '/');

    unsigned int nelem = elem.size();
    if(nelem != 4)
    {      
        LOG(fatal) << "GBnd::init"
                   << " bad boundary spec, expecting 4 elements"
                   << " spec " << spec
                   << " nelem " << nelem
                   ;

    }
    assert(nelem == 4);

    if(!flip_)
    {
        omat_ = elem[0].c_str() ;
        osur_ = elem[1].c_str() ;
        isur_ = elem[2].c_str() ;
        imat_ = elem[3].c_str() ;
    }
    else
    {
        omat_ = elem[3].c_str() ;
        osur_ = elem[2].c_str() ;
        isur_ = elem[1].c_str() ;
        imat_ = elem[0].c_str() ; 
    }

    omat = mlib->getIndex(omat_) ;
    osur = slib->getIndex(osur_) ;
    isur = slib->getIndex(isur_) ;
    imat = mlib->getIndex(imat_) ;
}



void GBnd::check()
{
    bool osur_unknown = has_osur() && osur == UNSET ; 
    bool isur_unknown = has_isur() && isur == UNSET ; 

    bool osur_unknown2 = has_osur() && slib->hasSurface(osur) == false ; 
    bool isur_unknown2 = has_isur() && slib->hasSurface(isur) == false ; 

    assert( osur_unknown == osur_unknown2  );
    assert( isur_unknown == isur_unknown2  );
    
    if(dbgbnd)
    {
        if(osur_unknown)
            LOG(error) << "[--dbgbnd]"
                       << " osur_unknown " << osur_
                       ;
        if(isur_unknown)
            LOG(error) << "[--dbgbnd]"
                       << " isur_unknown " << isur_ 
                       ;
    }
}





bool GBnd::has_isur() const 
{
    return isur_ && strlen(isur_) > 0 ; 
}
bool GBnd::has_osur() const 
{
    return osur_ && strlen(osur_) > 0 ; 
}



