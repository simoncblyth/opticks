#pragma once

#include <vector>
#include <string>

#include "BRAP_API_EXPORT.hh"
#include "BRAP_HEAD.hh"

/*
BPath
======

Parsing of the IDPATH geocache absolute path in various layouts.
This juicing of the IDPATH allows among other things the location 
of the src geometry file to be determined, avoiding the need to 
provide multiple paths.

*/

class BRAP_API BPath {
    public:
        BPath(const char* idpath);

        const char* getIdPath() const ; 
        const char* getIdFile() const ;   // eg g4_00.dae
        const char* getSrcDigest() const ; 
        const char* getIdName() const ;   // eg DayaBay_VGDX_20140414-1300
        const char* getGeoBase() const ; 
        const char* getPrefix() const ;   // eg /usr/local/opticks
        const char* getSrcPath() const ; 
        int         getLayout() const ; 

        std::string desc() const ; 

    private:
        void        init();
        void        parseLayout() ; 
        bool        isTriple(const char* triple ) const ;
        bool        parseTriple(const char* triple ) ;
        const char* getElem(int idx) const  ; 
    private:
        const char*              m_idpath ; 
        std::vector<std::string> m_elem ; 
        bool                     m_triple ; 
    private:
        const char*              m_idfile ;    // idfile is name of export file eg g4_00.dae
        const char*              m_srcdigest ; // 32-char hexdigest 
        const char*              m_idname  ;   // idname is name of dir containing the srcpath eg DayaBay_VGDX_20140414-1300
        const char*              m_geobase  ; 
        const char*              m_prefix  ; 
        const char*              m_srcpath  ; 
        int                      m_layout ; 
};

#include "BRAP_TAIL.hh"
 
