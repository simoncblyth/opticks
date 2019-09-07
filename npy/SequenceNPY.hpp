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

#include "NGLM.hpp"

#include <map>
#include <string>
#include <vector>

#include "Types.hpp"
#include "Counts.hpp"

template <typename T> class NPY ; 

class RecordsNPY ; 
class Index ; 

//
// precise agreement between Photon and Record histories
// demands setting a bounce max less that maxrec
// in order to avoid any truncated and top record slot overwrites 
//
// eg for maxrec 10 bounce max of 9 (option -b9) 
//    succeeds to give perfect agreement  
//                 

#include "NPY_API_EXPORT.hh"

class NPY_API SequenceNPY {
   public:  
       enum {
              e_seqhis , 
              e_seqmat 
            };
   public:  
       SequenceNPY(NPY<float>* photons); 
   public:  
       void                  setTypes(Types* types);
       void                  setRecs(RecordsNPY* recs);
   public:  
       NPY<float>*           getPhotons();
       RecordsNPY*           getRecs();
       Types*                getTypes();
   public:  
       Index*                makeHexIndex(const char* itemtype);
   public:  
       void                  indexSequences(unsigned int maxidx=32);
   public:  
       void                  dumpUniqueHistories();
       void                  countMaterials();
   public:  
       void                  setSeqIdx(NPY<unsigned char>* seqidx);
       NPY<unsigned char>*   getSeqIdx();
       Index*                getSeqHis(); 
       Index*                getSeqHisHex(); 
       Index*                getSeqMat(); 
   public:  
       NPY<unsigned long long>*  getSeqHisNpy(); 

   private:
       static bool second_value_order(const std::pair<int,int>&a, const std::pair<int,int>&b);
       static bool su_second_value_order(const std::pair<std::string,unsigned int>&a, const std::pair<std::string,unsigned int>&b);

   private:
        NPY<unsigned long long>* makeSequenceCountsArray( 
             Types::Item_t etype, 
             std::vector< std::pair<std::string, unsigned int> >& vp
              ); 

        Index* makeSequenceCountsIndex(
               Types::Item_t etype, 
               std::vector< std::pair<std::string, unsigned int> >& vp,
               unsigned long maxidx=32, 
               bool hex=false
               );

       void fillSequenceIndex(
                unsigned int k,
                Index* idx, 
                std::map<std::string, std::vector<unsigned int> >&  sv 
                );

       void dumpMaskCounts(
                const char* msg, 
                Types::Item_t etype, 
                std::map<unsigned int, unsigned int>& uu, 
                unsigned int cutoff
                );

       void dumpSequenceCounts(
                const char* msg, 
                Types::Item_t etype, 
                std::map<std::string, unsigned int>& su,
                std::map<std::string, std::vector<unsigned int> >& sv,
                unsigned int cutoff
                );

   private:
       NPY<float>*                  m_photons ; 
       RecordsNPY*                  m_recs ; 
       Types*                       m_types ; 
       NPY<unsigned char>*          m_seqidx  ; 
       unsigned int                 m_maxrec ; 
   private:
       Index*                       m_seqhis ; 
       Index*                       m_seqhis_hex ; 
       NPY<unsigned long long>*     m_seqhis_npy ; 
   private:
       Index*                       m_seqmat ; 
   private:
       Counts<unsigned int>         m_material_counts ; 
       Counts<unsigned int>         m_history_counts ; 

};



