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
#include "plog/Severity.h"

class BTimes ; 

#include "BRAP_API_EXPORT.hh"
#include "BRAP_HEAD.hh"

/**

BTimesTable
===========

A vector of labelled "columns" each of which holds a *BTimes* instance.  
The table can be saved/loaded to/from a directory where it is stored 
as individual ".ini" files named after the column labels. 

*BTimeKeeper* is the canonical user of *BTimesTable*

::

    simon:1 blyth$ BTimesTableTest 
    2016-09-15 11:11:08.829 INFO  [342255] [BTimesTable::dump@43] BTimesTable::dump
         t_absolute        t_delta
              5.944          5.944 : _seqhisMakeLookup
              5.951          0.007 : seqhisMakeLookup
              5.951          0.000 : seqhisApplyLookup
              5.951          0.000 : _seqmatMakeLookup
              5.956          0.005 : seqmatMakeLookup
              5.956          0.000 : seqmatApplyLookup
              5.986          0.030 : indexSequenceInterop
              6.025          0.039 : indexBoundaries
              6.028          0.003 : indexPresentationPrep
              6.137          0.110 : _save
              6.333          0.196 : save

**/

class BRAP_API BTimesTable {
    public:
        static const plog::Severity LEVEL ; 
        static const unsigned WIDTH ; 
        static const unsigned PRIME ; 
    public:
        BTimesTable(const char* columns, const char* delim=","); 
        BTimesTable(const std::vector<std::string>& columns);
        void dump(const char* msg="BTimesTable::dump", const char* startswith=NULL, const char* spacewith=NULL, double tcut=-1.0 );

        unsigned getNumColumns();
        BTimes* getColumn(unsigned int j);

        template <typename T> void add( T row, double x, double y, double z, double w, int count=-1 );
        template <typename T> const char* makeLabel( T row_, int count=-1 );

        std::vector<std::string>& getLines(); 

    public:
        void save(const char* dir);
        void load(const char* dir);
        const char* getLabel();

    private:
        void makeLines();
        void init(const std::vector<std::string>& columns);
        void setLabel(const char* label);

        unsigned getNumRows() const ; 
        const char* getColumnLabel(unsigned j) const ;
        std::string getColumnLabels() const ; 
    private:
        BTimes*   m_tx ; 
        BTimes*   m_ty ; 
        BTimes*   m_tz ; 
        BTimes*   m_tw ; 
        const char* m_label ; 

        std::vector<BTimes*>     m_table ; 
        std::vector<std::string> m_lines ; 
        std::vector<std::string> m_names ; 
        std::vector<double>      m_first ; 
};

#include "BRAP_TAIL.hh"




 
