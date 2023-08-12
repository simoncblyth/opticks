#pragma once
/**
scsg.hh : Manages pools of CSG nodes, param, bounding box and transforms
============================================================================

Canonical *csg* instance is instanciated by stree::stree and "snd::SetPOOL(csg)"
is called from stree::init. 

Note that the *idx* "integer pointer" used in the API corresponds to the 
relevant species node/param/aabb/xform for each method. 

CAUTION: DO NOT RETAIN POINTERS/REFS WHILST ADDING NODES 
AS REALLOC WILL SOMETIMES INVALIDATE THEM

Users of scsg.hh
-----------------

::

    epsilon:sysrap blyth$ opticks-f scsg.hh 
    ./sysrap/CMakeLists.txt:    scsg.hh
    ./sysrap/snd.hh:Usage requires the scsg.hh POOL. That is now done at stree instanciation::
    ./sysrap/stree.h:#include "scsg.hh"
    ./sysrap/snd.cc:#include "scsg.hh"
    ./sysrap/tests/snd_test.cc:#include "scsg.hh"
    ./sysrap/scsg.hh:scsg.hh : pools of CSG nodes, param, bounding box and transforms
    ./sysrap/scsg.cc:#include "scsg.hh"
    ./u4/tests/U4SolidTest.cc:#include "scsg.hh"
    epsilon:opticks blyth$ 


TODO : CAN THIS GO HEADER-ONLY .h ?
-------------------------------------

* snd.hh static pools a bit problematic  

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
    std::vector<spa<double>> param ; 
    std::vector<sbb<double>> aabb ; 
    std::vector<sxf<double>> xform ; 

    static constexpr const unsigned IMAX = 1000 ;

    scsg(); 
    void init(); 

    template<typename T>
    int add_(const T& obj, std::vector<T>& vec); 

    int addND(const snd& nd );
    int addPA(const spa<double>& pa );
    int addBB(const sbb<double>& bb );
    int addXF(const sxf<double>& xf );

    template<typename T>
    const T* get(int idx, const std::vector<T>& vec) const ; 

    const snd* getND(int idx) const ; 
    const spa<double>* getPA(int idx) const ;  
    const sbb<double>* getBB(int idx) const ;
    const sxf<double>* getXF(int idx) const ; 

    template<typename T>
    T* get_(int idx, std::vector<T>& vec) ; 

    snd* getND_(int idx); 
    spa<double>* getPA_(int idx);  
    sbb<double>* getBB_(int idx);
    sxf<double>* getXF_(int idx); 

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


    // slightly less "mechanical" methods than the above 
    void getLVID( std::vector<snd>& nds, int lvid ) const ; 
    const snd* getLVRoot(int lvid ) const ; 

};

