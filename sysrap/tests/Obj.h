#pragma once

#include "s_pool.h"
#include <iomanip>

struct Obj ; 

/**
_Obj : persistable type must not have any pointers 
**/
struct _Obj
{
    int type ;  
    int left ; 
    int right ; 
    int parent ; 

    std::string desc() const ; 
};
inline std::string _Obj::desc() const 
{
    std::stringstream ss ; 
    ss << "_Obj::desc " 
       << " type " << std::setw(5) << type 
       << " left " << std::setw(5) << left 
       << " right " << std::setw(5) << right 
       << " parent " << std::setw(5) << parent 
       ; 
    std::string str = ss.str(); 
    return str ; 
}


/**
Obj : ctor/dtor instrumented with pool.add(this)/pool.remove(this) 
**/

struct Obj 
{
    typedef s_pool<Obj, _Obj> POOL ;
    static POOL* pool ;

    static int Level(); 
    static Obj* Lookup(  int pid);  
    static Obj* GetByIdx(int idx);  
    static int Index(const Obj* o);
  
    int idx() const ; 
    std::string desc() const ; 

    Obj( int type, Obj* left=nullptr, Obj* right=nullptr ); 
    ~Obj();  

    static std::string Desc(const Obj* o); 
    static void Serialize(    _Obj& p, const Obj* o ); 
    static Obj* Import( const _Obj* p, const std::vector<_Obj>& buf ); 
    static Obj* Import_r( const _Obj* p, const std::vector<_Obj>& buf ); 

    int   pid ;   // HMM: not required, but useful for debug   
    int   type ; 
    Obj*  left ; 
    Obj*  right ; 
    Obj*  parent ; 
};

inline int  Obj::Level() {  return ssys::getenvint("Obj__level",-1) ; } // static 
inline Obj* Obj::Lookup(  int pid){ return pool->lookup(pid) ; } // static
inline Obj* Obj::GetByIdx(int idx){ return pool->getbyidx(idx) ; } // static
inline int Obj::Index(const Obj* o){ return pool ? pool->index(o) : -1 ; } // static

inline int Obj::idx() const { return Index(this); } 


inline std::string Obj::desc() const
{
    std::stringstream ss ;
    ss << "Obj::desc"
       << " pid " << pid
       << " type " << type 
       ; 
    std::string str = ss.str(); 
    return str ; 

}

inline Obj::Obj( int type_, Obj* left_, Obj* right_ )
    :
    pid(pool->add(this)),
    type(type_),
    left(left_),
    right(right_)
{
    if(Level()>1) std::cout << "[ Obj::Obj pid " << pid << "\n" ; 

    if( left && right )
    {
        left->parent = this ; 
        right->parent = this ; 
    }

    if(Level()>1) std::cout << "] Obj::Obj pid " << pid << "\n" ; 
}

/**
Obj dtor
----------

As Obj does recursive deletion it means that should be creating Obj 
onto heap (not stack) for control, 
using things like std::vector<Obj> will cause double dtors 

**/

inline Obj::~Obj()
{
    if(Level()>1) std::cout << "[ Obj::~Obj pid " << pid << "\n" ; 
    delete left ; 
    delete right ; 

    pool->remove(this) ; 
    if(Level()>1) std::cout << "] Obj::~Obj pid " << pid << "\n" ; 
}


inline std::string Obj::Desc(const Obj* o)
{
    return o ? o->desc() : "-" ; 
}


/**
Obj::Serialize
-----------------
 
**/

inline void Obj::Serialize( _Obj& p, const Obj* o ) // static
{
    p.type   = o->type ; 
    p.left   = pool->index(o->left);  
    p.right  = pool->index(o->right);  
    p.parent = pool->index(o->parent);  

    std::cerr << "Obj::Serialize p " << p.desc() << std::endl ; 
}
/**
Obj::Import
-----------------

HMM: when the Obj are arranged into one or more trees 
its easiest to import recursively by the roots, so need
a way to identify the roots from the persisted _Obj type.  
Simple way to do that is with parent links that are -1 
for roots. 
 
**/

inline Obj* Obj::Import( const _Obj* p, const std::vector<_Obj>& buf ) // static
{
    std::cerr << "Obj::Import " << p->desc() << std::endl ; 
    Obj* root = nullptr ; 
    if(p->parent == -1) root = Import_r(p, buf); 
    return root ; 
}

/**
Obj::Import_r
--------------

Separating recursive parts of the import from the root parts is clearer.

**/

inline Obj* Obj::Import_r( const _Obj* p, const std::vector<_Obj>& buf )
{
    if(p == nullptr) return nullptr ; 

    std::cerr << "Obj::Import_r " << p->desc() << std::endl ; 

    int type = p->type ;  
    const _Obj* _left  = p->left  > -1 ? &buf[p->left]  : nullptr ;  
    const _Obj* _right = p->right > -1 ? &buf[p->right] : nullptr ;  

    Obj* left  = Import_r( _left , buf ); 
    Obj* right = Import_r( _right, buf ); 
    Obj* node  = new Obj( type, left, right ); 
    return node ; 
}


