#pragma once
/**
scsg.hh : pools of CSG nodes, param, bounding box and transforms
==================================================================


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
    int level ; 
    std::vector<snd> node ; 
    std::vector<spa> param ; 
    std::vector<sbb> aabb ; 
    std::vector<sxf> xform ; 

    static constexpr const unsigned IMAX = 1000 ;

    scsg(); 
    void init(); 

    template<typename T>
    int add_(const T& obj, std::vector<T>& vec); 

    int addND(const snd& nd );
    int addPA(const spa& pa );
    int addXF(const sxf& xf );
    int addBB(const sbb& bb );

    // CAUTION: DO NOT RETAIN POINTERS/REFS WHILST ADDING NODES 
    // AS REALLOC WILL SOMETIMES INVALIDATE THEM

    template<typename T>
    const T* get(int idx, const std::vector<T>& vec) const ; 

    // *idx* "pointer" corresponding to each species, nothing clever here
    const snd* getND(int idx) const ; 
    const spa* getPA(int idx) const ;  
    const sbb* getBB(int idx) const ;
    const sxf* getXF(int idx) const ; 

    const glm::tmat4x4<double>* getXForm(int idx) const ; 


    template<typename T>
    T* get_(int idx, std::vector<T>& vec) ; 

    snd* getND_(int idx); 
    spa* getPA_(int idx);  
    sbb* getBB_(int idx);
    sxf* getXF_(int idx); 

    glm::tmat4x4<double>* getXForm_(int idx); 


    int  getNDXF(int idx) const ; 
    void getLVID( std::vector<snd>& nds, int lvid ) const ; 
    const snd* getLVRoot(int lvid ) const ; 


    template<typename T>
    std::string desc_(int idx, const std::vector<T>& vec) const ; 

    std::string descND(int idx) const ; 
    std::string descPA(int idx) const ; 
    std::string descBB(int idx) const ;  
    std::string descXF(int idx) const ; 

    std::string brief() const ; 
    std::string desc() const ; 

    NPFold* serialize() const ; 
    void    import(const NPFold* fold); 
};

