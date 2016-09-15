#pragma once

#include <vector>
#include <string>

class Times ; 

#include "NPY_API_EXPORT.hh"
#include "NPY_HEAD.hh"

/**

TimesTable
===========

A vector of labelled "columns" each of which 
holds a *Times* instance. 

::

    simon:1 blyth$ TimesTableTest 
    2016-09-15 11:11:08.829 INFO  [342255] [TimesTable::dump@43] TimesTable::dump
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

class NPY_API TimesTable {
    public:
        TimesTable(const char* columns, const char* delim=","); 
        TimesTable(const std::vector<std::string>& columns);
        void dump(const char* msg="TimesTable::dump");

        unsigned getNumColumns();
        Times* getColumn(unsigned int j);

        template <typename T> void add( T row, double x, double y, double z, double w, int count=-1 );
        template <typename T> const char* makeLabel( T row_, int count=-1 );

        void makeLines();
        std::vector<std::string>& getLines(); 
    public:
        void save(const char* dir);
        void load(const char* dir);
    private:
        void init(const std::vector<std::string>& columns);
    private:
        Times*   m_tx ; 
        Times*   m_ty ; 
        Times*   m_tz ; 
        Times*   m_tw ; 

        std::vector<Times*>      m_table ; 
        std::vector<std::string> m_lines ; 
};

#include "NPY_TAIL.hh"




 
