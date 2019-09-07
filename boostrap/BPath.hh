/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */

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

Examples:

layout 0
     /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae

     has dotted triple as last element: g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae
     which yields idfile (eg g4_00.dae) and digest (96ff965744a2f6b78c24e33c80d3a4cd) 

layout > 0
     /usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/1 

      last elem is the layout integer


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
        bool        isDigest(const char* last ) const ;
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
 
