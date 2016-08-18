#pragma once

#include <string>

#include "BRAP_API_EXPORT.hh"
#include "BRAP_HEAD.hh"

class BRAP_API  BOpticksEvent {
        friend class BOpticksEventTest ; 
   public:
        static const int DEFAULT_LAYOUT_VERSION ; 
        static int       LAYOUT_VERSION ; 
        static const char* DEFAULT_DIR_TEMPLATE  ;
        static const char* OVERRIDE_EVENT_BASE ;
   public:
        virtual void Summary(const char* msg="BOpticksEvent::Summary");
   public:
       static void SetLayoutVersion(int version);  
       static void SetLayoutVersionDefault();  
       static void SetOverrideEventBase(const char* override_event_base); // NB remember to clear override by setting NULL after use

       // formerly in NPYBase, moved to lower level for wider availability 
       //static std::string directory(const char* tfmt, const char* targ, const char* det);
       //static std::string directory(const char* typ, const char* det);
       //static std::string path(const char* pfx, const char* gen, const char* tag, const char* det );
       //static std::string path(const char* typ, const char* tag, const char* det );

   public:
       static std::string directory(const char* top, const char* sub, const char* tag);
       static std::string path(     const char* top, const char* sub, const char* tag, const char* stem, const char* ext=".npy");
       static std::string path(const char* dir, const char* name);
   private:
       static std::string directory_template();
       static std::string directory_(const char* top, const char* sub, const char* tag=".");
       static std::string path_(const char* top, const char* sub, const char* tag, const char* stem, const char* ext=".npy");

};

#include "BRAP_TAIL.hh"

