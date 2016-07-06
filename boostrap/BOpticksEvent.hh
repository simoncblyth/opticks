#pragma once

#include <string>

#include "BRAP_API_EXPORT.hh"
#include "BRAP_HEAD.hh"

class BRAP_API  BOpticksEvent {
   public:
        static const char* DEFAULT_DIR_TEMPLATE  ;
   public:
        BOpticksEvent();
        virtual ~BOpticksEvent();
        virtual void Summary(const char* msg="BOpticksEvent::Summary");
   public:
       // formerly in NPYBase, moved to lower level for wider availability 
       static std::string directory(const char* tfmt, const char* targ, const char* det);
       static std::string directory(const char* typ, const char* det);

       static std::string path(const char* dir, const char* name);
       static std::string path(const char* typ, const char* tag, const char* det);
       static std::string path(const char* pfx, const char* gen, const char* tag, const char* det);

   private:
        void init();
   protected:
};

#include "BRAP_TAIL.hh"

