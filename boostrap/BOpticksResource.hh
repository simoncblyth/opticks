#pragma once

#include "BRAP_API_EXPORT.hh"
#include "BRAP_HEAD.hh"

class BRAP_API  BOpticksResource {
   public:
        BOpticksResource(const char* envprefix="OPTICKS_");
        virtual ~BOpticksResource();
        virtual void Summary(const char* msg="BOpticksResource::Summary");
   private:
        void init();
        void adoptInstallPrefix();
        void setTopDownDirs();
        //void readG4Environment();
        //void readOpticksEnvironment();
   protected:
        const char* m_envprefix ; 
        const char* m_install_prefix ;   // from BOpticksResourceCMakeConfig header
        const char* m_opticksdata_dir ; 
        const char* m_resource_dir ; 
};

#include "BRAP_TAIL.hh"

