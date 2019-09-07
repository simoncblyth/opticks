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
#include <vector>


#include "BRAP_API_EXPORT.hh"
#include "BRAP_HEAD.hh"
#include "plog/Severity.h"

class BTxt ; 


class BRAP_API BLog {
   public:
       static const plog::Severity LEVEL ; 
       static const double TOLERANCE ; 

       static const char* VALUE ; 
       static const char* CUT ; 
       static const char* NOTE ; 

       static const char* DELIM ; 
       static const char* END ; 

       static BLog* Load(const char* path); 
       static int ParseKV( const std::string& line,  const char* start, const char* delim, const char* end, std::string& k, std::string& v ); 
       static int Compare( const BLog* a , const BLog* b ) ; 
   public:
       BLog(); 

       void setSequence(const std::vector<double>*  sequence);   

       void addValue( const char* key,  double value ); 
       int getIdx() const ; 

       void addCut(   const char* ckey, double cvalue ); 
       void addNote(  const char* nkey, int nvalue ); 

       unsigned    getNumKeys() const ;  
       const char* getKey(unsigned i) const ; 
       double      getValue(unsigned i) const ; 
       int         getSequenceIndex(unsigned i ) const ;
       const std::vector<double>&  getValues() const ; 

       void        dump(const char* msg="BLog::dump") const ; 


       std::string  makeLine(unsigned i) const ; 

       std::string  makeValueString(unsigned i, bool present=false) const ; 
       std::string  makeCutString(  unsigned i, bool present=false) const ; 
       std::string  makeNoteString (unsigned i, bool present=false) const ; 

       BTxt*        makeTxt() const ; 
       void         write(const char* path) const ; 

   private:
       void init();  

   private:
       std::vector<std::string>     m_keys ; 
       std::vector<double>          m_values ; 


       typedef std::pair<int, std::string>   PIS_t ; 
       typedef std::pair<int, double>        PID_t ; 
       typedef std::pair<int, int>           PII_t ; 

       std::vector<PIS_t> m_ckeys ;   
       std::vector<PID_t> m_cvalues ;   

       std::vector<PIS_t> m_nkeys ;   
       std::vector<PII_t> m_nvalues ;   

       const std::vector<double>*   m_sequence ;    


};

#include "BRAP_TAIL.hh"

