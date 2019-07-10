#pragma once

#include <vector>
#include <string>
class BTimesTable ; 
template <typename T> class NPY ; 

#include "OKCORE_API_EXPORT.hh"
#include "OKCORE_HEAD.hh"
#include "plog/Severity.h"

/**

OpticksProfile
===============

Recording time and virtual memory as various points during Opticks running.


**/


class OKCORE_API OpticksProfile 
{
    private:
       static const plog::Severity LEVEL ; 
       static const char* NAME ; 
    public:
       static OpticksProfile* Load( const char* dir); 
    public:
       OpticksProfile();
       template <typename T> void stampOld(T row, int count);
       void stamp(const char* label, int count);

       void setStamp(bool stamp); 

       std::vector<std::string>&  getLines(); 
       void save();
       void load();
       void dump(const char* msg="OpticksProfile::dump", const char* startswith=NULL, const char* spacewith=NULL, double tcut=0.0);

       void setDir(const char* dir);
       const char* getDir();
       const char* getName();
       std::string getPath();

       std::string brief();

    public:
       void save(const char* dir);
    private:
       void setT(float t);
       void setVM(float vm);
       void load(const char* dir);
    private:
       bool        m_stamp ;  
       const char* m_dir ; 
       const char* m_name ; 
       const char* m_lname ; 
       const char* m_columns ; 
       BTimesTable* m_tt ; 

       NPY<float>* m_npy ;
       NPY<char>*  m_lpy ;
 
       float       m_t0 ; 
       float       m_tprev ; 
       float       m_t ; 

       float       m_vm0 ; 
       float       m_vmprev ; 
       float       m_vm ; 

       unsigned    m_num_stamp ; 

};

#include "OKCORE_TAIL.hh"



