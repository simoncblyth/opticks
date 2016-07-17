#include <cassert>
#include <cstring>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <boost/algorithm/string/replace.hpp>


#include "BFile.hh"
#include "BOpticksEvent.hh"

#include "PLOG.hh"

const char* BOpticksEvent::OVERRIDE_EVENT_BASE = NULL ; 


BOpticksEvent::BOpticksEvent()
{
    init();
}

BOpticksEvent::~BOpticksEvent()
{
}

void BOpticksEvent::init()
{
}

void BOpticksEvent::Summary(const char* msg)
{
    LOG(info) << msg ; 

}

const char* BOpticksEvent::DEFAULT_DIR_TEMPLATE = "$OPTICKS_EVENT_BASE/evt/$1/$2" ;  // formerly "$LOCAL_BASE/env/opticks/$1/$2"

std::string BOpticksEvent::path(const char* pfx, const char* gen, const char* tag, const char* det)
{
    std::stringstream ss ;
    ss << pfx << gen ;
    std::string typ = ss.str() ;
    return path(typ.c_str(), tag, det);
}

std::string BOpticksEvent::directory(const char* tfmt, const char* targ, const char* det)
{
    char typ[64];
    if(strchr (tfmt, '%' ) == NULL)
    {
        snprintf(typ, 64, "%s%s", tfmt, targ ); 
    }
    else
    { 
        snprintf(typ, 64, tfmt, targ ); 
    }
    std::string dir = directory(typ, det);
    return dir ; 
}


void BOpticksEvent::SetOverrideEventBase(const char* override_event_base)
{
   OVERRIDE_EVENT_BASE = override_event_base ? strdup(override_event_base) : NULL ; 
}


std::string BOpticksEvent::directory(const char* typ, const char* det)
{
    std::string deftmpl(DEFAULT_DIR_TEMPLATE) ; 
    if(OVERRIDE_EVENT_BASE)
    {
       LOG(info) << "BOpticksEvent::directory OVERRIDE_EVENT_BASE replacing OPTICKS_EVENT_BASE with " << OVERRIDE_EVENT_BASE ; 
       boost::replace_first(deftmpl, "$OPTICKS_EVENT_BASE/evt", OVERRIDE_EVENT_BASE );
    } 

    boost::replace_first(deftmpl, "$1", det );
    boost::replace_first(deftmpl, "$2", typ );
    std::string dir = BFile::FormPath( deftmpl.c_str() ); 
    return dir ;
}
  
std::string BOpticksEvent::path(const char* typ, const char* tag, const char* det)
{

// :param typ: object type name, eg oxcerenkov rxcerenkov 
// :param tag: event tag, usually numerical 
// :param det: detector tag, eg dyb, juno

    std::string dir = directory(typ, det);
    dir += "/%s.npy" ; 

    char* tmpl = (char*)dir.c_str();
    char path_[256];
    snprintf(path_, 256, tmpl, tag );

    LOG(debug) << "BOpticksEvent::path"
              << " typ " << typ
              << " tag " << tag
              << " det " << det
              << " DEFAULT_DIR_TEMPLATE " << DEFAULT_DIR_TEMPLATE
              << " OVERRIDE_EVENT_BASE " << ( OVERRIDE_EVENT_BASE ? OVERRIDE_EVENT_BASE : "NULL" )
              << " tmpl " << tmpl
              << " path_ " << path_
              ;

    return path_ ;   
}

std::string BOpticksEvent::path(const char* dir, const char* name)
{
    char path[256];
    snprintf(path, 256, "%s/%s", dir, name);

    //std::string path = BFile::FormPath(dir, name);  
    // provides native style path with auto-prefixing based on envvar  
    return path ; 
}



