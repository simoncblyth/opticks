#include "scsg.hh"
#include "ssys.h"
#include "NPFold.h"
#include "NPX.h"

scsg::scsg()
    :
    level(ssys::getenvint("scsg_level", 0))
{
    init(); 
}

/**
scsg::init
------------

Reserving suffiencient IMAX prevents realloc of the vectors,
however it is better to code in a way to avoid the need to 
reserve in order to avoid issues happening once the total 
number of nodes exceeds the IMAX.   

DO THIS BY NOT USING POINTERS OR REFERNCES OBTAINED
BEFORE PUSH_BACK TO A VECTOR AFTER THE PUSH_BACK.
THAT MEANS DO THE Add LAST THING IN METHODS
AND DO NOT STORE POINTERS OR REFERENCES

**/

void scsg::init()
{
   /*
    node.reserve(IMAX);   
    param.reserve(IMAX);   
    xform.reserve(IMAX);   
    aabb.reserve(IMAX);   
   */ 
}


template<typename T>
int scsg::add_(const T& obj, std::vector<T>& vec) 
{
    int idx = vec.size(); 
    vec.push_back(obj);  
    return idx ; 
}

template<>
int scsg::add_(const snd& obj, std::vector<snd>& vec) 
{
    int idx = vec.size(); 

    const snd* ptr0_before = &vec[0] ; 
    vec.push_back(obj); 
    const snd* ptr0_after = &vec[0] ; 
    
    if( ptr0_before != ptr0_after && level > 0) 
    {
        std::cerr << "scsg::add_<snd> DETECTED REALLOC AT idx " << idx << std::endl ;  
    }

    vec[idx].index = idx ;   // template specialization for snd : record the index 
    return idx  ; 
}


int scsg::addND(const snd& nd) { return add_<snd>(nd, node)  ; }
int scsg::addPA(const spa& pa) { return add_<spa>(pa, param) ; }
int scsg::addXF(const sxf& xf) { return add_<sxf>(xf, xform) ; }
int scsg::addBB(const sbb& bb) { return add_<sbb>(bb, aabb)  ; } 


template<typename T>
const T* scsg::get(int idx, const std::vector<T>& vec) const  
{
    assert( idx < int(vec.size()) ); 
    return idx < 0 ? nullptr : &vec[idx] ; 
}
const snd* scsg::getND(int idx) const { return get<snd>(idx, node)  ; }
const spa* scsg::getPA(int idx) const { return get<spa>(idx, param) ; }
const sxf* scsg::getXF(int idx) const { return get<sxf>(idx, xform) ; }
const sbb* scsg::getBB(int idx) const { return get<sbb>(idx, aabb)  ; } 





template<typename T>
T* scsg::get_(int idx, std::vector<T>& vec) 
{
    assert( idx < int(vec.size()) ); 
    return idx < 0 ? nullptr : &vec[idx] ; 
}

snd* scsg::getND_(int idx) { return get_<snd>(idx, node)  ; }
spa* scsg::getPA_(int idx) { return get_<spa>(idx, param) ; }
sxf* scsg::getXF_(int idx) { return get_<sxf>(idx, xform) ; }
sbb* scsg::getBB_(int idx) { return get_<sbb>(idx, aabb)  ; } 


int scsg::getNDXF(int idx) const  
{
    const snd* n = getND(idx); 
    return n ? n->xform : -1 ; 
}

void scsg::getLVID( std::vector<snd>& nds, int lvid ) const 
{
    int num_node = node.size(); 
    for(int i=0 ; i < num_node ; i++)
    {
        const snd& nd = node[i] ; 
        if(nd.lvid == lvid) nds.push_back(nd) ; 
    }
}



template<typename T>
std::string scsg::desc_(int idx, const std::vector<T>& vec) const   
{
    int w = 3 ; 
    int num_obj = vec.size() ; 

    std::stringstream ss ; 
    ss << T::NAME << ":" << std::setw(w) << idx << " " ;  
    if(idx < 0) 
    {
        ss << "(none)" ; 
    }
    else if( idx >= num_obj )
    {
        ss << "(invalid)" ; 
    }
    else
    {
        const T& obj = vec[idx] ;  
        ss << obj.desc() ; 
    }
    std::string str = ss.str(); 
    return str ; 
}


std::string scsg::descND(int idx) const { return desc_<snd>(idx, node); }
std::string scsg::descPA(int idx) const { return desc_<spa>(idx, param); }
std::string scsg::descBB(int idx) const { return desc_<sbb>(idx, aabb); }
std::string scsg::descXF(int idx) const { return desc_<sxf>(idx, xform); }





std::string scsg::brief() const
{
    std::stringstream ss ; 
    ss << "scsg" 
       << " node " << node.size() 
       << " param " << param.size() 
       << " aabb " << aabb.size() 
       << " xform " << xform.size() 
       ;
    std::string str = ss.str(); 
    return str ; 
}
std::string scsg::desc() const 
{
    std::stringstream ss ; 
    ss << brief() << std::endl ; 
    for(int idx=0 ; idx < int(node.size()) ; idx++) ss << descND(idx) << std::endl ; 
    std::string str = ss.str(); 
    return str ; 
}


NPFold* scsg::serialize() const 
{
    NPFold* fold = new NPFold ; 
    fold->add("node",  NPX::ArrayFromVec<int,    snd>(node)); 
    fold->add("param", NPX::ArrayFromVec<double, spa>(param)); 
    fold->add("aabb",  NPX::ArrayFromVec<double, sbb>(aabb)); 
    fold->add("xform", NPX::ArrayFromVec<double, sxf>(xform)); 
    return fold ; 
}
void scsg::import(const NPFold* fold) 
{ 
    NPX::VecFromArray<snd>(node,  fold->get("node"));  // NB the vec are cleared first 
    NPX::VecFromArray<spa>(param, fold->get("param")); 
    NPX::VecFromArray<sbb>(aabb,  fold->get("aabb")); 
    NPX::VecFromArray<sxf>(xform, fold->get("xform")); 
}


