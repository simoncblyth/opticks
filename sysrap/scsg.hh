#pragma once
/**
scsg.hh
=========

pools that are index referenced from the snd instances 

**/

#include <vector>
#include "SYSRAP_API_EXPORT.hh"

#include "snd.hh"
#include "spa.h"
#include "sbb.h"
#include "sxf.h"

struct NPFold ; 


struct SYSRAP_API scsg
{
    std::vector<snd>   node ; 
    std::vector<spa>   param ; 
    std::vector<sbb>   aabb ; 
    std::vector<sxf>   xform ; 

    template<typename T>
    int add_(const T& obj, std::vector<T>& vec); 

    int addNode( const snd& nd );
    int addParam(const spa& pa );
    int addXForm(const sxf& xf );
    int addAABB( const sbb& bb );


    template<typename T>
    const T* get_(int idx, const std::vector<T>& vec) const ; 

    // CAUTION: DO NOT RETAIN POINTERS THEY MAY GO STALE
    const snd* getNode( int idx) const ; 
    const spa* getParam(int idx) const ;  
    const sxf* getXForm(int idx) const ; 
    const sbb* getAABB( int idx) const ;

    int  getNodeXForm(int idx) const ; 

    std::string brief() const ; 
    std::string desc() const ; 
    NPFold* serialize() const ; 
    void    import(const NPFold* fold); 

    template<typename T>
    std::string desc_(int idx, const std::vector<T>& vec) const ; 

    std::string descND(int idx) const ; 
    std::string descPA(int idx) const ; 
    std::string descBB(int idx) const ;  
    std::string descXF(int idx) const ; 
};



