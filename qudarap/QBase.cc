#include <sstream>

#include "SSys.hh"
#include "PLOG.hh"
#include "QU.hh"

#include "QBase.hh"
#include "qbase.h"

const plog::Severity QBase::LEVEL = PLOG::EnvLevel("QBase", "DEBUG"); 
const QBase* QBase::INSTANCE = nullptr ; 
const QBase* QBase::Get(){ return INSTANCE ; }

qbase* QBase::MakeInstance() // static 
{
    qbase* base = new qbase ; 
    base->pidx = SSys::getenvint("PIDX", -1) ; 
    return base ; 
}

QBase::QBase()
    :
    base(MakeInstance()),
    d_base(QU::UploadArray<qbase>(base,1))
{
}

std::string QBase::desc() const 
{
    std::stringstream ss ; 
    ss << "QBase::desc"
       << " base " << base 
       << " d_base " << d_base 
       << " base.desc " << base->desc()
       ; 
    std::string s = ss.str(); 
    return s ; 
}



