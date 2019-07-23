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

Canonical m_profile instance is resident of Opticks and
is instanciated with it.



Recording time and virtual memory as various points during Opticks running.

Ideas to improve stopwatch 

* https://codereview.stackexchange.com/questions/196245/extremely-simple-timer-class-in-c



**/

struct OKCORE_API OpticksAcc
{
    unsigned n ; 
    float    t ; 
    float    v ; 
    float    t0 ; 
    float    v0 ; 

    static void Init(OpticksAcc& acc )
    { 
        acc.n = 0 ; 
        acc.t = 0.f ; 
        acc.v = 0.f ; 
        acc.t0 = 0.f ; 
        acc.v0 = 0.f ;
    }  
};

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

       unsigned accumulateAdd(const char* label); 
       void     accumulateStart(unsigned idx); 
       void     accumulateStop(unsigned idx); 
       std::string accumulateDesc(unsigned idx) const ;
       void     accumulateDump(const char* msg) const ;
       void     accumulateExport()  ;
       bool     isAccExported() const ;

       void setStamp(bool stamp); 

       std::vector<std::string>&  getLines(); 
       void save();
       void load();
       void dump(const char* msg="OpticksProfile::dump", const char* startswith=NULL, const char* spacewith=NULL, double tcut=0.0);

       template<typename T>
       void setMeta(const char* key, T value ); 

       template<typename T>
       T getMeta(const char* key, const char* fallback) const ; 


       void setDir(const char* dir);
       const char* getDir() const ;
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
       const char* m_aname ; 
       const char* m_laname ; 

       const char* m_columns ; 
       BTimesTable* m_tt ; 

       NPY<float>* m_npy ;
       NPY<char>*  m_lpy ;
       NPY<float>* m_apy ;
       NPY<char>*  m_lapy ;
 
       float       m_t0 ; 
       float       m_tprev ; 
       float       m_t ; 

       float       m_vm0 ; 
       float       m_vmprev ; 
       float       m_vm ; 

       unsigned    m_num_stamp ; 

       std::vector<OpticksAcc>   m_acc ; 
       std::vector<std::string>  m_acc_labels  ; 

};

#include "OKCORE_TAIL.hh"



