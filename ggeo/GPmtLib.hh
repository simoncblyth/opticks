#pragma once

#include <string>

class Opticks ; 
class GBndLib ; 
class GPmt ; 
class GMergedMesh ; 

#include "GGEO_API_EXPORT.hh"
#include "GGEO_HEAD.hh"

/**

GPmtLib
==========

DIRTY ASSOCIATION BETWEEN OLD STYLE ANALYTIC GPmt AND TRIANGULATED GMergedMesh 
    
GPmt 
  detdesc parsed analytic geometry (see pmt-ecd dd.py tree.py etc..)
    

**/

class GGEO_API GPmtLib {
        friend class CTestDetector ;  // for getLoadedAnalyticPmt
    public:
        //void save();
        static const char* TRI_PMT_PATH ; 
        static GPmtLib* load(Opticks* ok, GBndLib* bndlib);
    public:
        GPmtLib(Opticks* ok, GBndLib* bndlib);
        GMergedMesh* getPmt() ;
    private:
        GPmt* getLoadedAnalyticPmt();
    private:
        void loadAnaPmt();
        void loadTriPmt();
        void dirtyAssociation();
        std::string getTriPmtPath();
    private:
        Opticks*     m_ok ; 
        GBndLib*     m_bndlib ; 
    private:
        GPmt*        m_apmt ; 
        GMergedMesh* m_tpmt ; 
 
};

#include "GGEO_TAIL.hh"

