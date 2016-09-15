#pragma once

#include <string>
class TimesTable ; 
template <typename T> class NPY ; 

#include "OKCORE_API_EXPORT.hh"
#include "OKCORE_HEAD.hh"

class OKCORE_API OpticksProfile 
{
    public:
       OpticksProfile(const char* dir="$TMP", const char* name="OpticksProfile");
       template <typename T> void stamp(T row);
       void save();
       void load();
       void dump(const char* msg="OpticksProfile::dump");
       const char* getName();
       const char* getDir();
       std::string brief();
    private:
       void setT(float t);
       void setVM(float vm);
       void save(const char* dir);
       void load(const char* dir);
    private:
       const char* m_dir ; 
       const char* m_name ; 
       const char* m_columns ; 
       TimesTable* m_tt ; 
       NPY<float>* m_npy ;
 
       float       m_t0 ; 
       float       m_tprev ; 
       float       m_t ; 

       float       m_vm0 ; 
       float       m_vmprev ; 
       float       m_vm ; 

       unsigned    m_count ; 

};

#include "OKCORE_TAIL.hh"



