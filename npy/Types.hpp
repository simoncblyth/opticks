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

#include <string>
#include <map>
#include <vector>

#include "glm/fwd.hpp"

class Index ; 



/*

//
// this is on the way out 
// moving to GPropertyLib GAttrSeq approach divvying 
// out labelling into handlers from GFlags, GMaterialLib, GSurfaceLib, ...
//

Places to migrate to Typ::

    simon:opticks blyth$ opticks-find-typ
    ./ggeo/GGeo.cc:#include "Typ.hpp"
    ./ggeo/tests/RecordsNPYTest.cc:#include "Typ.hpp"
    ./optickscore/OpticksResource.cc:#include "Typ.hpp"

    simon:opticks blyth$ opticks-find-types
    ./ggeo/GItemIndex.cc:#include "Types.hpp"
    ./ggeo/tests/GItemIndexTest.cc:#include "Types.hpp"
    ./ggeo/tests/RecordsNPYTest.cc:#include "Types.hpp"
    ./oglrap/OpticksViz.cc:#include "Types.hpp"
    ./oglrap/Photons.cc:#include "Types.hpp"
    ./optickscore/OpticksResource.cc:#include "Types.hpp"
    ./opticksgeo/OpticksIdx.cc:#include "Types.hpp"
    ./opticksnpy/tests/_BoundariesNPYTest.cc:#include "Types.hpp"
    ./opticksnpy/tests/_RecordsNPYTest.cc:#include "Types.hpp"
    ./opticksnpy/tests/PhotonsNPYTest.cc:#include "Types.hpp"
    ./opticksnpy/tests/SequenceNPYTest.cc:#include "Types.hpp"
    ./opticksnpy/tests/TypesTest.cc:#include "Types.hpp"
    simon:opticks blyth$ 

*/



#include "NPY_API_EXPORT.hh"
#include "NPY_HEAD.hh"

class NPY_API Types {
   public:
       static const char* ENUM_HEADER_PATH ;  // NB duplicitous with OpticksFlags 
   public:
       static const char* TAIL ; 

       static const char* HISTORY_ ; 
       static const char* MATERIAL_ ; 
       static const char* HISTORYSEQ_ ; 
       static const char* MATERIALSEQ_ ; 

       typedef enum { HISTORY, MATERIAL, HISTORYSEQ, MATERIALSEQ } Item_t ;

   public:
       Types(); 
   private:
       void init();
   public:
       void         setTail(const char* tail);
       const char*  getTail();
       void         setAbbrev(bool abbrev);
   public:
       Index*       getFlagsIndex();
       Index*       getMaterialsIndex();
       void         setMaterialsIndex(Index* index);
   public:
       std::string getMaskString(unsigned int mask, Item_t etype);
       std::string getMaterialString(unsigned int flags);
       std::string getHistoryString(unsigned int flags);
       unsigned int getHistoryFlag(std::string label);
       unsigned int getMaterialCode(std::string label);
       std::string getSequenceString(unsigned long long seq);
       const char* getItemName(Item_t item);

   public:
       void getMaterialStringTest();
       void readMaterialsOld(const char* idpath, const char* name="GMaterialIndexLocal.json");    
       void readMaterials(const char* idpath, const char* name="GMaterialIndex", const char* reldir=NULL);    
       void dumpMaterials(const char* msg="Types::dumpMaterials");
       std::string findMaterialName(unsigned int index);
   private:
       void makeMaterialAbbrev();

   public:
       std::string getStepFlagString(unsigned char flag);
       void getHistoryStringTest();

       void readFlags(const char* path); // parse enum flags from photon.h
       void dumpFlags(const char* msg="Types::dumpFlags");
       void saveFlags(const char* idpath, const char* ext=".ini");
   private:
       void makeFlagAbbrev();
   public:
       glm::ivec4                getFlags();
       bool*                     initBooleanSelection(unsigned int n);

       // used from Photons::gui_flag_selection for ImGui::Checkbox
       std::vector<std::string>& getFlagLabels();
       bool*                     getFlagSelection();
   public:
       std::string getHistoryAbbrev(std::string label);
       std::string getMaterialAbbrev(std::string label);
       std::string getAbbrev(std::string label, Item_t etype);
   public:
       std::string getHistoryAbbrevInvert(std::string label, bool hex=false);
       std::string getMaterialAbbrevInvert(std::string label, bool hex=false);
       std::string getAbbrevInvert(std::string label, Item_t etype, bool hex=false);
       unsigned int getHistoryAbbrevInvertAsCode(std::string label, bool hex=false);
       unsigned int getMaterialAbbrevInvertAsCode(std::string label, bool hex=false);
       unsigned int getAbbrevInvertAsCode(std::string label, Item_t etype, bool hex=false);

   public:
       // formerly in RecordsNPY
       unsigned long long convertSequenceString(std::string& seq, Item_t etype, bool hex=false);
       void prepSequenceString(std::string& lseq, unsigned int& elen, unsigned int& nelem, bool hex);
       std::string decodeSequenceString(std::string& seq, Item_t etype, bool hex=false);

       // used by GItemIndex::materialSeqLabeller GItemIndex::historySeqLabeller
       std::string abbreviateHexSequenceString(std::string& seq, Item_t etype);

   private:
       Index*                                               m_materials_index ; 
       std::map<std::string, unsigned int>                  m_materials ;
       std::map<std::string, std::string>                   m_material2abbrev ; 
       std::map<std::string, std::string>                   m_abbrev2material ; 

   private:
       Index*                                               m_flags ; 
       std::vector<std::string>                             m_flag_labels ; 
       std::vector<unsigned int>                            m_flag_codes ; 
       bool*                                                m_flag_selection ; 
   private:
       std::map<std::string, std::string>                   m_flag2abbrev ; 
       std::map<std::string, std::string>                   m_abbrev2flag ; 

   private:
       const char*  m_tail ;
       bool         m_abbrev ; 

};

#include "NPY_TAIL.hh"


