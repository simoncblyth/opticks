#pragma once

#include <string>

#include "BRAP_API_EXPORT.hh"
#include "BRAP_HEAD.hh"

/**
BOpticksEvent
===============

Standardized paths for storing Opticks Events based on tag etc...
This functionality was formerly in NPYBase, it was moved to lower level 
for wider availability .

Used widely, opticks-findl BOpticksEvent.hh (ordered by relevance)::

    ./optickscore/OpticksEvent.cc
    ./optickscore/OpticksEventSpec.cc
    ./npy/NPYBase.cpp
    ./npy/NLoad.cpp
    ./boostrap/BOpticksEvent.cc

    ./ggeo/tests/GItemIndexTest.cc
    ./boostrap/tests/BOpticksEventTest.cc
    ./optixrap/tests/LTOOContextUploadDownloadTest.cc
    ./npy/tests/NPYTest.cc
    ./boostrap/CMakeLists.txt

**/

class BRAP_API  BOpticksEvent {
        friend class BOpticksEventTest ; 
   public:
        static const int DEFAULT_LAYOUT_VERSION ; 
        static int       LAYOUT_VERSION ; 
        static const char* DEFAULT_DIR_TEMPLATE  ;
        static const char* DEFAULT_DIR_TEMPLATE_NOTAG  ;
        static const char* DEFAULT_DIR_TEMPLATE_RELATIVE  ;
        static const char* OVERRIDE_EVENT_BASE ;
   public:
        virtual void Summary(const char* msg="BOpticksEvent::Summary");
   public:
       static void SetLayoutVersion(int version);  
       static void SetLayoutVersionDefault();  
       static void SetOverrideEventBase(const char* override_event_base); // NB remember to clear override by setting NULL after use

   public:
       static std::string reldir(const char* top, const char* sub, const char* tag );
       static std::string directory(const char* top, const char* sub, const char* tag, const char* anno=NULL  );
       static std::string path(     const char* top, const char* sub, const char* tag, const char* stem, const char* ext=".npy");
       static const char* srctagdir( const char* det, const char* typ, const char* tag);
       
       //static std::string path(const char* dir, const char* name);
       //static std::string path(const char* dir, const char* reldir, const char* name);
   private:
       static std::string directory_template(bool notag=false);
       static std::string directory_(const char* top, const char* sub, const char* tag, const char* anno=NULL );
       static std::string path_(const char* top, const char* sub, const char* tag, const char* stem, const char* ext=".npy");

};

#include "BRAP_TAIL.hh"

