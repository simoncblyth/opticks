#pragma once
/**
NPX.h : NP.hh related extras such as static converters
=========================================================


**/

#include "NP.hh"
#include <unordered_map>


struct NPX
{
    template<typename T>
    static NP* MakeValues( const std::vector<std::string>& keys, const std::vector<T>& vals ); 
    template<typename T>
    static NP* MakeValues( const std::vector<std::pair<std::string, T>>& values, const char* contains=nullptr ); 
    template<typename T>
    static std::string DescValues(const NP* a); 

    template<typename T>
    struct KV
    {
        std::vector<std::string> kk ; 
        std::vector<T> vv ; 

        void add(const char* k, T v ){ kk.push_back(k); vv.push_back(v); }
        NP* values() const { return MakeValues<T>(kk, vv) ; }
    }; 


    static NP* MakeDemo(const char* dtype="<f4" , NP::INT ni=-1, NP::INT nj=-1, NP::INT nk=-1, NP::INT nl=-1, NP::INT nm=-1, NP::INT no=-1 ); 

    template<typename T> static NP*  Make( const std::vector<T>& src ); 
    template<typename T> static NP*  Make( T d0, T v0, T d1, T v1 ); 
    template<typename T, typename... Args> static NP*  Make(const T* src, Args ... shape );  // WIP: switch to name ArrayFromData
    template<typename T, typename... Args> static NP*  ArrayFromData(const T* src, Args ... shape ); 

    template<typename T> static NP* FromString(const char* str, char delim=' ') ;  


    static NP* Holder( const std::vector<std::string>& names ); 


    template<typename T, int N>
    static NP* ArrayFromVecOfArrays(const std::vector<std::array<T,N>>& va );

    template<typename T, int N>
    static void VecOfArraysFromArray( std::vector<std::array<T,N>>& va, const NP* a );



    template<typename T, typename S> 
    static NP* ArrayFromVec__(const std::vector<S>& v, const std::vector<NP::INT>& itemshape ); 

    template<typename T, typename S, typename... Args> 
    static NP* ArrayFromVec(const std::vector<S>& v, Args ... args_itemshape );   // ArrayFromVec_ellipsis

    template<typename T, typename S> 
    static NP* ArrayFromVec_(const std::vector<S>& v, const char* str_itemshape );  


    template<typename S> 
    static void VecFromArray(std::vector<S>& v, const NP* a );

    template<typename S>
    static int VecFromMap( std::vector<S>& v,  const std::map<int, S>& m, bool contiguous_key=true ); 

    template<typename S>
    static void VecFromMapUnordered( 
        std::vector<S>& v,  
        const std::unordered_map<int, S>& m, 
        const std::vector<int>* key_order, 
        bool* all_contiguous_key,
        int* _k0,
        int* _k1
        ); 


    template<typename S>
    static void MapFromVec( std::map<int, S>& m,  const std::vector<S>& v, int k0=0, bool contiguous_key=true ); 

    template<typename S>
    static void MapUnorderedFromVec( std::unordered_map<int, S>& m,  const std::vector<S>& v, int kmin=0, bool is_contiguous_key=true ); 


    template<typename T, typename S>
    static NP* ArrayFromMap( const std::map<int, S>& m, bool contiguous_key=true ); 

    template<typename T, typename S>
    static NP* ArrayFromMapUnordered( 
        const std::unordered_map<int, S>& m, const std::vector<int>* key_order=nullptr ); 

    template<typename S>
    static void KeyRangeMapUnordered( int* _k0, int* _k1, const std::unordered_map<int, S>& m );

    template<typename S>
    static void MapFromArray( std::map<int, S>& m, const NP* a ); 

    template<typename S>
    static void MapUnorderedFromArray( std::unordered_map<int, S>& m, const NP* a ); 


    template<typename S>
    static NP* ArrayFromDiscoMap( const std::map<int, S>& m ); 
    template<typename S>
    static void DiscoMapFromArray( std::map<int, S>& m, const NP* a ); 
    template<typename S>
    static std::string DescDiscoMap( const std::map<int, S>& m ); 


    template<typename S>
    static NP* ArrayFromDiscoMapUnordered( const std::unordered_map<int, S>& m ); 
    template<typename S>
    static void DiscoMapUnorderedFromArray( std::unordered_map<int, S>& m, const NP* a ); 
    template<typename S>
    static std::string DescDiscoMapUnordered( const std::unordered_map<int, S>& m ); 





    template<typename T> 
    static NP* FromNumpyString(const char* str) ;  

    static NP* CategoryArrayFromString(const char* str, int catfield, const char* cats, char delim=','); 
    static NP* LoadCategoryArrayFromTxtFile(const char* base, const char* relp, int catfield, const char* cats, char delim=',');
    static NP* LoadCategoryArrayFromTxtFile(const char* path, int catfield, const char* cats, char delim=',');

    static void Import_MSD(           std::map<std::string,double>& msd, const NP* a); 
    static NP* Serialize_MSD(   const std::map<std::string,double>& msd );  
    static std::string Desc_MSD(const std::map<std::string,double>& msd); 

    static NP* ArrayFromEnumMap( const std::map<int, std::string>& catMap) ; 

    static NP* MakeCharArray( const std::vector<std::string>& nn ); 

    template<typename F, typename T>
    static NP* BOA( NP* a, NP* b, NP::INT a_column=-1, NP::INT b_column=-1, std::ostream* out=nullptr  ); 

}; 



template<typename T>
inline NP* NPX::MakeValues( const std::vector<std::string>& keys, const std::vector<T>& vals ) // static
{
    assert( keys.size() == vals.size() ); 
    if(vals.size() == 0 ) return nullptr ; 

    NP* vv = NPX::Make<T>( vals ) ; 
    vv->set_names( keys );  
    return vv ; 
}

template<typename T>
inline NP* NPX::MakeValues( const std::vector<std::pair<std::string, T>>& values, const char* contains ) // static
{
    if(NP::VERBOSE) std::cout 
        << "NPX::MakeValues values.size " << values.size() 
        << " contains " << ( contains ? contains : "-" )
        << std::endl 
        ;  

    std::vector<std::string> nams ; 
    std::vector<T> vals ; 

    for(NP::INT i=0 ; i < NP::INT(values.size()) ; i++)
    {   
        const std::pair<std::string, T>& kv = values[i] ; 
        const char* k = kv.first.c_str() ; 
        T v = kv.second ;

        bool select = contains == nullptr || U::Contains( k, contains ) ; 

        if(NP::VERBOSE) std::cout 
            << "NPX::MakeValues " 
            << std::setw(3) << i 
            << " v " << std::setw(10) << std::fixed << std::setprecision(4) << v 
            << " k " << std::setw(60) << k 
            << " select " << select 
            <<  std::endl 
            ;
 
        if(select)
        {   
            nams.push_back(k); 
            vals.push_back(v); 
        }   
    }  
    if(NP::VERBOSE) std::cout << "NPX::MakeValues vals.size " << vals.size() << std::endl ;  

    NP* vv = MakeValues<T>(nams, vals); 
    return vv ; 
}

template<typename T>
inline std::string NPX::DescValues(const NP* a) // static
{
    std::stringstream ss ;
    ss << std::endl << "NPX::descValues"  << std::endl ; 
    for(unsigned i=0 ; i < a->names.size() ; i++)
    {
        const char* k = a->names[i].c_str(); 
        T v = a->get_named_value<T>(k, 0) ; 
        ss
            << std::setw(30) << k 
            << " : " 
            << std::setw(10) << v 
            << std::endl
            ; 
    } 
    std::string str = ss.str(); 
    return str ; 
}


inline NP* NPX::MakeDemo(const char* dtype, NP::INT ni, NP::INT nj, NP::INT nk, NP::INT nl, NP::INT nm, NP::INT no )
{
    NP* a = new NP(dtype, ni, nj, nk, nl, nm, no);
    a->fillIndexFlat(); 
    return a ; 
}



template <typename T> 
inline NP* NPX::Make( const std::vector<T>& src ) // static
{
    NP* a = NP::Make<T>(src.size()); 
    a->read(src.data()); 
    return a ; 
}

template <typename T> 
inline NP*  NPX::Make(T d0, T v0, T d1, T v1 ) // static
{
    std::vector<T> src = {d0, v1, d1, v1 } ; 
    return NPX::Make<T>(src) ; 
}

/**
NPX::Make "Make_ellipsis"
--------------------------

This "Make_ellipsis" method combines allocation of the array and populating it 
from the src data. This is intended to facilitate creating arrays from vectors
of struct, by using simple template types  (int, float, double etc.. )  
together with array item shapes appropriate to the elements of the struct. 
For example::

   struct demo { int x,y,z,w ; } ; 
   std::vector<demo> dd ; 
   dd.push_back( {1,2,3,4} ); 

   NP* a = NPX::Make<int>( (int*)dd.data() , int(dd.size()) , 4 ); 

The product of the shape integers MUST correspond to the number of 
values provided from the src data. 
When the first int shape dimension is zero a nullptr is returned.

**/

template<typename T, typename... Args> 
inline NP* NPX::Make(const T* src, Args ... args )   // WIP switch to name ArrayFromData
{
    std::cerr << "TODO: change NPX::Make to NPX::ArrayFromData \n" ; 
    return ArrayFromData(src, std::forward<Args>(args)...) ; 
}

template<typename T, typename... Args> 
inline NP* NPX::ArrayFromData(const T* src, Args ... args )   
{
    std::string dtype = descr_<T>::dtype() ; 
    std::vector<NP::INT> shape = {args...};
    if(shape.size() > 0 && shape[0] == 0) return nullptr ; 
    NP* a = new NP(dtype.c_str(), shape ); 
    a->read2(src);  
    return a ; 
}



template <typename T> 
inline NP* NPX::FromString(const char* str, char delim)  // static 
{   
    std::vector<T> vec ; 
    std::stringstream ss(str);
    std::string s ; 
    while(getline(ss, s, delim)) vec.push_back(U::To<T>(s.c_str()));
    NP* a = Make<T>(vec) ; 
    return a ; 
}







inline NP* NPX::Holder( const std::vector<std::string>& names )
{
    NP* a = NP::Make<int>(0) ; 
    a->set_names(names) ; 
    return a ; 
}




/**
NPX::ArrayFromVecOfArrays
--------------------------

There is potential for the impl of std::vector and std::array
to add padding so although the compound vector of arrays will be contiguous
it is not possible to rely on having the obvious layout ?

**/

template<typename T, int N>
inline NP* NPX::ArrayFromVecOfArrays(const std::vector<std::array<T,N>>& va )
{
    NP::INT ni = va.size();
    NP::INT nj = N ;
    NP* a = NP::Make<T>(ni, nj);
    T* aa = a->values<T>();
    for(NP::INT i=0 ; i < ni ; i++)
    {
        const std::array<T,N>& arr = va[i] ;
        for(NP::INT j=0 ; j < nj ; j++) aa[i*nj+j] = arr[j] ;
    }
    return a ;
}


template<typename T, int N>
inline void NPX::VecOfArraysFromArray( std::vector<std::array<T,N>>& va, const NP* a )
{
    assert( a && a->uifc == 'f' && a->ebyte == sizeof(T) );
    assert( a && a->shape.size() == 2 );

    NP::INT ni = a->shape[0];
    NP::INT nj = a->shape[1];
    const T* aa = a->cvalues<T>();

    va.resize(ni);
    for(NP::INT i=0 ; i < ni ; i++)
    {
        std::array<T,N>& arr = va[i] ;
        for(NP::INT j=0 ; j < nj ; j++) arr[j] = aa[i*nj+j] ;
    }

}






template<typename T, typename S> 
inline NP* NPX::ArrayFromVec__(const std::vector<S>& v, const std::vector<NP::INT>& itemshape )   
{
    assert( sizeof(S) >= sizeof(T) );  
    NP::INT ni = v.size() ; 
    NP::INT nj = sizeof(S) / sizeof(T) ; 

    const T* src = (T*)v.data() ; 

    std::vector<NP::INT> shape ; 
    shape.push_back(ni) ; 

    if(itemshape.size() == 0 )
    {
        shape.push_back(nj) ; 
    }
    else 
    {
        NP::INT itemcheck = 1 ; 
        for(NP::INT i=0 ; i < NP::INT(itemshape.size()) ; i++)  
        {
            shape.push_back(itemshape[i]) ; 
            itemcheck *= itemshape[i] ; 
        }
        bool consistent = itemcheck == nj ;

        if(!consistent) std::cerr 
            << "NPX::ArrayFromVec__"
            << " ERROR " 
            << " consistent " << ( consistent ? "YES" : "NO " )
            << " itemcheck " << itemcheck 
            << " nj " << nj 
            << " itemshape.size " << itemshape.size()
            << std::endl 
            ;  

        assert(consistent); 
    }


    NP* a = NP::Make_<T>(shape) ; 
    a->read2(src);  
    return a ; 
}




/**
NPX::ArrayFromVec
-------------------

The optional itemshape integers override the flat item element count 
obtained from the type sizeof ratio *sizeof(S)/sizeof(T)*.
Note that the product of the itemshape integers must match the 
flat item count however.  

**/

template<typename T, typename S, typename... Args> 
inline NP* NPX::ArrayFromVec(const std::vector<S>& v, Args ... args_itemshape )   // ArrayFromVec_ellipsis
{
    std::vector<NP::INT> itemshape = {args_itemshape...};
    return ArrayFromVec__<T,S>(v, itemshape ); 
}

template<typename T, typename S> 
inline NP* NPX::ArrayFromVec_(const std::vector<S>& v, const char* str_itemshape )
{
    std::vector<NP::INT> itemshape ; 
    U::MakeVec(itemshape,  str_itemshape, ',' ); 
    return ArrayFromVec__<T,S>(v, itemshape ); 
}



/**
NPX::VecFromArray
-------------------

Direct mapping assuming type S is correct. 

**/


template<typename S> 
inline void NPX::VecFromArray(std::vector<S>& v, const NP* a )
{
   if(a == nullptr || a->shape.size() == 0 ) return ; 
   NP::INT ni = a->shape[0] ; 
   unsigned ib = a->item_bytes() ;   
   bool expected_sizeof_item = sizeof(S) == ib  ; 

   if(!expected_sizeof_item) 
      std::cerr 
          << "NPX::VecFromArray"
          << " expected_sizeof_item  " << ( expected_sizeof_item  ? "YES" : "NO" )
          << " sizeof(S) " << sizeof(S) 
          << " a.item_bytes " << ib
          << " a.sstr " << a->sstr()
          << std::endl 
          << " a.lpath " << a->get_lpath()
          << std::endl 
          << " CHECK FOR COMPILATION OPTIONS THAT CHANGE STRUCT SIZES "
          << std::endl 
          << " FOR EXAMPLE WITH_CHILD CHANGES sysrap/sn.h "
          << std::endl
          << " ANOTHER POSSIBILITY IS LOADING AN ARRAY WRITTEN BEFORE STRUCT SIZE CHANGES "
          << std::endl
          ;

   assert( expected_sizeof_item ); 

   v.clear() ; 
   v.resize(ni); 

   memcpy( v.data(), a->bytes(), a->arr_bytes() ); 
}



/**
NPX::VecFromMap
---------------

Populate vector using map values in the default key order of the map 

1. clear and resize the vector to match size of the map 
2. record k0, first key in map
3. iterate through the map copying S values from the map into the vector
4. return k0 

contiguous_key:true
   an assert verifies that keys are contiguously incrementing from the first key

Used by NPX::ArrayFromMap

**/


template<typename S>
inline int NPX::VecFromMap( std::vector<S>& v,  const std::map<int, S>& m, bool contiguous_key ) // static
{
    NP::INT ni = NP::INT(m.size()) ; 

    v.clear(); 
    v.resize(ni); 

    typename std::map<int,S>::const_iterator it = m.begin()  ;
    int k0 = it->first ; 

    for(NP::INT idx=0 ; idx < ni ; idx++)
    {   
        int k = it->first ; 
        v[idx] = it->second ;

        if(contiguous_key) assert( k == k0 + idx );  
        //std::cout << " k0 " << k0 << " idx " << idx << " k " << k << " v " << it->second << std::endl ;  

        std::advance(it, 1);  
    }   
    return k0 ; 
}

/**
NPX::VecFromMapUnordered
--------------------------

Populate vector using unordered_map values using key_order vector ordering.

When the key_order argument is not nullptr it must point to 
a vector that contains all the keys that are present in the
unordered map in the desired order. 
When the key_order argument is nullptr the keys are assumed to 
be contiguous integers starting from the minimum key that 
is present in the unordered_map.  

1. clear vector and resize vector to the size of the map
2. when key_order points to a vector iterate over keys, for each key:

   * count if the key is a contiguous offset from the firsy key, k0
   * use the key to lookup a value from the map and place that value into the idx vector position
   * NB all keys in the unorderd_map must be present in the key_order vector,
     otherwise an unordedmap::at failure will occur

3. when key_order is nullptr determine the range of keys and use the smallest key *_k0
   as the base for all keys, that are assumed to be contiguous.  If the unorderd_map keys 
   are not within a contiguous range of integers then this method 
   will fail with an unordedmap::at failure. In this situation it is necesssary 
   to provide a pointer to a key_order vector contaning all the keys in the unordered_map.

**/

template<typename S>
inline void NPX::VecFromMapUnordered( 
    std::vector<S>& v,  
    const std::unordered_map<int, S>& m, 
    const std::vector<int>* key_order, 
    bool* all_contiguous_key,
    int* _k0, 
    int* _k1 
    )
{
    NP::INT ni = m.size()  ; 
    v.clear(); 
    v.resize(ni); 

    if(key_order)
    {
        assert( key_order->size() == m.size() ) ; 
    }

    NP::INT count(0) ; 
    if( key_order == nullptr ) KeyRangeMapUnordered(_k0, _k1, m ) ; 


    NP::INT k0_check(-999) ; 

    for(NP::INT idx=0 ; idx < ni ; idx++)
    {   
        NP::INT k = key_order ? (*key_order)[idx] : *_k0 + idx  ; 

        // counting keys that are contiguous offsets from the idx 0 key  
        if( idx == 0 ) 
        {
            k0_check = k ; 
            count += 1 ; 
        } 
        else
        {
            if( k == k0_check + idx ) count += 1  ;  
        } 

        const S& s = m.at(k); 
        v[idx] = s ; 
    }
 
    *all_contiguous_key = ni > 0 && count == ni  ; 

    if(key_order == nullptr) 
    {
        assert( *all_contiguous_key ) ; 
        assert( *_k1 - *_k0 + 1 == ni ) ; 
        assert( k0_check == *_k0 ); 
    }

}





/**
NPX::MapFromVec
----------------

HMM: to support contiguous_key:false would need type S to follow 
some convention such as keeping int keys within the first 32 bit member. 
OR just store keys and values as separate arrays within an NPFold.

**/

template<typename S>
inline void NPX::MapFromVec( std::map<int, S>& m,  const std::vector<S>& v, int k0, bool contiguous_key )
{
    assert( contiguous_key == true ); 

    NP::INT ni = NP::INT(v.size()) ; 
    m.clear(); 

    for(NP::INT i=0 ; i < ni ; i++)
    {
        const S& item = v[i] ; 
        NP::INT key = k0 + i ;  
        m[key] = item ; 
    }
}




/**
NPX::MapUnorderedFromVec
---------------------------


**/

template<typename S>
inline void NPX::MapUnorderedFromVec( std::unordered_map<int, S>& m,  const std::vector<S>& v, int kmin, bool is_contiguous_key )
{
    assert( is_contiguous_key == true ); 

    NP::INT ni = NP::INT(v.size()) ; 
    m.clear(); 

    for(NP::INT idx=0 ; idx < ni ; idx++)
    {
        const S& item = v[idx] ; 
        NP::INT key = kmin + idx ;  
        m[key] = item ; 
    }
}









/**
NPX::ArrayFromMap
-------------------

A vector of S structs is populated from the map in the default key order of the map. 
An NP array is then created from the contiguous vector data.  

When contiguous_key:true the map keys are required to contiguously increment
from the first. The first key is recorded into the metadata of the array with name "k0". 
For example with keys: 100,101,102 the k0 would be 100. 

Serializing maps is most useful for contiguous_key:true as 
map access by key can then be mimicked by simply obtaining the 
array index by subtracting k0 from the map key.  

**/

template<typename T, typename S>
inline NP* NPX::ArrayFromMap( const std::map<int, S>& m, bool contiguous_key )
{
    assert( sizeof(S) >= sizeof(T) );

    std::vector<S> v ;    
    int k0 = NPX::VecFromMap<S>( v, m, contiguous_key ); 
    NP* a = NPX::ArrayFromVec<T,S>(v) ;

    int ContiguousKey = contiguous_key ? 1 : 0 ; 

    if(NP::VERBOSE) std::cout 
       << "NPX::ArrayFromMap"
       << " k0 " << k0
       << " ContiguousKey " << ContiguousKey
       << std::endl
       ;

    a->set_meta<int>("k0", k0) ;
    a->set_meta<int>("ContiguousKey", ContiguousKey) ;

    return a ;
}


/**
NPX::ArrayFromMapUnordered
-----------------------------

Only useful when the keys are contiguous 
as key values not persisted. 

See also NPX::MapUnorderedFromArray that 
does the reverse, recreating the unordered_map 
from the array. 

**/


template<typename T, typename S>
inline NP* NPX::ArrayFromMapUnordered( const std::unordered_map<int, S>& m, const std::vector<int>* key_order )
{
    assert( sizeof(S) >= sizeof(T) );

    std::vector<S> v ;    
    bool all_contiguous_key(false) ; 
    int kmin(0) ; 
    int kmax(0) ; 

    NPX::VecFromMapUnordered<S>( v, m, key_order, &all_contiguous_key, &kmin, &kmax ); 
    NP* a = NPX::ArrayFromVec<T,S>(v) ;

    a->set_meta<int>("kmin", kmin) ;
    a->set_meta<int>("kmax", kmax) ;
    a->set_meta<int>("ContiguousKey", all_contiguous_key ) ;
    a->set_meta<std::string>("Creator", "NPX::ArrayFromMapUnordered" ); 

    return a ;
}



template<typename S>
inline void NPX::KeyRangeMapUnordered( int* _k0, int* _k1, const std::unordered_map<int, S>& m ) 
{
    assert( _k0 ); 
    assert( _k1 ); 

    *_k0 = std::numeric_limits<int>::max() ; 
    *_k1 = std::numeric_limits<int>::min() ;
 
    typedef std::unordered_map<int,S> UIS ; 
    for(typename UIS::const_iterator it=m.begin() ; it != m.end() ; it++ ) 
    {
        int k = it->first ; 
        if( k < *_k0 ) *_k0 = k ; 
        if( k > *_k1 ) *_k1 = k ; 
    }
}







template<typename S>
inline void NPX::MapFromArray( std::map<int, S>& m, const NP* a )
{
    if(a == nullptr || a->shape.size() == 0 ) return ; 

    int k0 = a->get_meta<int>("k0"); 
    int ContiguousKey = a->get_meta<int>("ContiguousKey") ; 
    if(NP::VERBOSE) std::cout 
        << "NPX::MapFromArray"
        << " k0 " << k0
        << " ContiguousKey " << ContiguousKey
        << std::endl 
        ;

    std::vector<S> v ;    
    NPX::VecFromArray<S>(v, a); 
    
    NPX::MapFromVec(m, v, k0, ContiguousKey == 1 ); 
}



template<typename S>
inline void NPX::MapUnorderedFromArray( std::unordered_map<int, S>& m, const NP* a )
{
    if(a == nullptr || a->shape.size() == 0 ) return ; 

    int kmin = a->get_meta<int>("kmin"); 
    int kmax = a->get_meta<int>("kmax"); 
    int ContiguousKey = a->get_meta<int>("ContiguousKey") ; 
    bool is_contiguous_key = ContiguousKey == 1 ;  


    if(NP::VERBOSE) std::cout 
        << "NPX::MapUnorderedFromArray"
        << " kmin " << kmin
        << " kmax " << kmax
        << " is_contiguous_key " << ( is_contiguous_key ? "YES" : "NO " ) 
        << std::endl 
        ;

    std::vector<S> v ;    
    NPX::VecFromArray<S>(v, a); 
    
    NPX::MapUnorderedFromVec<S>(m, v, kmin, is_contiguous_key ); 
}




/**
NPX::ArrayFromDiscoMap
------------------------

How to handle maps with discontiguous keys ? 
For S:int and int keys can simply save as array of shape (10,2) 

  idx  key val 
   0    0   0
   1    1   1
   2    2   2
   3    3   0
   4    4   1
   5    5   2
   6   10   3
   7   11   3
   8   12   3
   9   13   3

**/

template<typename S>
inline NP* NPX::ArrayFromDiscoMap( const std::map<int, S>& m )
{
    return nullptr ;    
}

template<typename S>
inline NP* NPX::ArrayFromDiscoMapUnordered( const std::unordered_map<int, S>& m )
{
    return nullptr ;    
}





template<>
inline NP* NPX::ArrayFromDiscoMap( const std::map<int,int>& m )
{
    NP::INT ni = m.size() ; 
    NP::INT nj = 2 ; 
    NP* a = NP::Make<int>(ni, nj) ;  
    int* aa = a->values<int>(); 

    typedef std::map<int,int> MII ; 
    MII::const_iterator it = m.begin(); 

    for(NP::INT i=0 ; i < ni ; i++)
    {
        aa[i*nj+0] = it->first ;  
        aa[i*nj+1] = it->second ;  
        it++ ; 
    }
    return a ; 
}


/**
NPX::ArrayFromDiscoMapUnordered
---------------------------------

1. collect keys from the unordered_map
2. sort the keys into ascending order
3. interate through the sorted keys lookinhg up values from the unordered_map
   and populating the array 

**/

template<>
inline NP* NPX::ArrayFromDiscoMapUnordered( const std::unordered_map<int,int>& m )
{
    std::vector<int> keys ; 
    typedef std::unordered_map<int,int> MII ; 
    for(MII::const_iterator it=m.begin() ; it != m.end() ; it++) keys.push_back(it->first); 
    std::sort(keys.begin(), keys.end());

    NP::INT ni = m.size() ; 
    assert( NP::INT(keys.size()) == ni ); 
 
    NP::INT nj = 2 ; 
    NP* a = NP::Make<int>(ni, nj) ;  
    int* aa = a->values<int>(); 

    for(NP::INT i=0 ; i < ni ; i++)
    {
        NP::INT key = keys[i] ; 
        int val = m.at(key) ; 
        aa[i*nj+0] = key ;  
        aa[i*nj+1] = val ;  
    }
    return a ; 
}











template<typename S>
inline void NPX::DiscoMapFromArray( std::map<int, S>& m, const NP* a ){}

template<typename S>
inline void NPX::DiscoMapUnorderedFromArray( std::unordered_map<int, S>& m, const NP* a ){}




template<>
inline void NPX::DiscoMapFromArray( std::map<int, int>& m, const NP* a )
{
    assert( a && a->uifc == 'i' && a->ebyte == 4 && a->shape.size() == 2 ); 
    NP::INT ni = a->shape[0] ; 
    NP::INT nj = a->shape[1] ;
    assert( nj == 2 );  

    const int* aa = a->cvalues<int>(); 
    for(NP::INT i=0 ; i < ni ; i++)
    {
        int k = aa[i*nj+0] ;  
        int v = aa[i*nj+1] ;  
        m[k] = v ;  
    }
}


template<>
inline void NPX::DiscoMapUnorderedFromArray( std::unordered_map<int, int>& m, const NP* a )
{
    assert( a && a->uifc == 'i' && a->ebyte == 4 && a->shape.size() == 2 ); 
    NP::INT ni = a->shape[0] ; 
    NP::INT nj = a->shape[1] ;
    assert( nj == 2 );  

    const int* aa = a->cvalues<int>(); 
    for(NP::INT i=0 ; i < ni ; i++)
    {
        int k = aa[i*nj+0] ;  
        int v = aa[i*nj+1] ;  
        m[k] = v ;  
    }
}










template<typename S>
inline std::string NPX::DescDiscoMap( const std::map<int, S>& m )
{
    std::stringstream ss ; 
    ss << "NPX::DescDiscoMap" << std::endl << " m.size " << m.size() ; 
    std::string s = ss.str();    
    return s ; 
}

template<typename S>
inline std::string NPX::DescDiscoMapUnordered( const std::unordered_map<int, S>& m )
{
    std::stringstream ss ; 
    ss << "NPX::DescDiscoMapUnordered" << std::endl << " m.size " << m.size() ; 
    std::string s = ss.str();    
    return s ; 
}




template<>
inline std::string NPX::DescDiscoMap( const std::map<int,int>& m )
{
    NP::INT ni = m.size() ; 
    typedef std::map<int,int> MII ; 
    MII::const_iterator it = m.begin(); 
    std::stringstream ss ; 
    ss << "NPX::DescDiscoMap" << std::endl << " m.size " << ni << std::endl ; 
    for(NP::INT i=0 ; i < ni ; i++)
    {
        ss << "( " << it->first << " : " << it->second << " ) " << std::endl ;   
        it++ ; 
    }
    std::string s = ss.str();    
    return s ; 
}


template<>
inline std::string NPX::DescDiscoMapUnordered( const std::unordered_map<int,int>& m )
{
    NP::INT ni = m.size() ; 
    typedef std::unordered_map<int,int> MII ; 
    MII::const_iterator it = m.begin(); 
    std::stringstream ss ; 
    ss << "NPX::DescDiscoMapUnordered" << std::endl << " m.size " << ni << std::endl ; 
    for(NP::INT i=0 ; i < ni ; i++)
    {
        ss << "( " << it->first << " : " << it->second << " ) " << std::endl ;   
        it++ ; 
    }
    std::string s = ss.str();    
    return s ; 
}












template <typename T> 
inline NP* NPX::FromNumpyString(const char* str)  // static 
{   
    std::vector<T> vec ; 
    std::stringstream fss(str);
    bool dump = false ; 

    int num_field_0 = 0 ; 
    std::string line ; 
    while(getline(fss, line))
    {
        if(strlen(line.c_str())==0) continue ; 
        if(dump) std::cout << "{" << line << "}" << std::endl ; 
        std::istringstream iss(line);
        std::vector<std::string> fields ; 
        std::string field ; 
        while( iss >> field ) fields.push_back(field) ;

        int num_field = 0 ; 

        if(dump) std::cout << "fields.size " << fields.size() << std::endl ; 
        for(NP::INT j=0 ; j < NP::INT(fields.size()) ; j++ )
        {
           const char* fld = fields[j].c_str() ;  
           char* fldd = U::FirstToLastDigit(fld); 
           if(fldd == nullptr) continue ; 

           T val = U::To<T>(fldd) ; 

           if(dump) std::cout 
               << "{" << fld << "}" 
               << "{" << fldd << "}" 
               << "{" << val << "}" 
               <<  std::endl
               ;  
           num_field += 1 ; 
           vec.push_back( val ); 
        }
        if( num_field_0 == 0 ) 
        {
            num_field_0 = num_field ; 
        }
        else
        {
            assert( num_field_0 == num_field ); 
        }
    }

    NP* a = Make<T>(vec) ; 
    a->change_shape(-1, num_field_0);  

    return a ; 
}

inline NP* NPX::CategoryArrayFromString(const char* str, int catfield, const char* cats_, char delim )
{
    std::vector<std::string> cats ; 
    U::MakeVec(cats, cats_, delim );  

    int num_field = 0 ; 
    std::vector<int> data ; 
    std::string line ; 
    std::stringstream fss(str) ;
    while(std::getline(fss, line)) 
    {        
        std::istringstream iss(line);
        std::vector<std::string> fields ; 
        std::string field ; 
        while( iss >> field ) fields.push_back(field) ;
            
        if(num_field == 0) num_field = fields.size() ; 
        else  assert( int(fields.size()) == num_field ); // require consistent number of fields

        assert( catfield < num_field );  
        for(int i=0 ; i < num_field ; i++)
        {   
            const std::string& fld = fields[i] ; 
            int val =  i == catfield  ? U::Category(cats, fld ) : std::atoi(fld.c_str()) ;   
            data.push_back(val); 
        }   
    }   

    NP* a = Make<int>( data );  
    a->change_shape(-1, num_field); 
    a->set_names(cats); 
    return a ; 
} 

inline NP* NPX::LoadCategoryArrayFromTxtFile(const char* base, const char* relp, int catfield, const char* cats, char delim  )  // static
{
    std::string path = U::form_path(base, relp); 
    return LoadCategoryArrayFromTxtFile(path.c_str(), catfield, cats, delim) ; 
}
inline NP* NPX::LoadCategoryArrayFromTxtFile(const char* path, int catfield, const char* cats, char delim  )  // static 
{   
    const char* str = U::ReadString2(path); 
    if(str == nullptr) return nullptr ; 
    NP* a = CategoryArrayFromString(str, catfield, cats, delim ); 
    return a ; 
}

inline void NPX::Import_MSD( std::map<std::string, double>& msd, const NP* a) // static
{
    assert( a && a->uifc == 'f' && a->ebyte == 8 );
    assert( a->shape.size() == 1 );
    assert( NP::INT(a->names.size()) == a->shape[0] );

    const double* vv = a->cvalues<double>() ;
    unsigned num_vals = a->shape[0] ;

    for(unsigned i=0 ; i < num_vals ; i++)
    {
        const std::string& key = a->names[i] ;
        const double& val = vv[i] ;
        msd[key] = val ;
    }
}

inline NP* NPX::Serialize_MSD( const std::map<std::string, double>& msd ) // static
{
    typedef std::map<std::string, double> MSD ; 
    MSD::const_iterator it = msd.begin(); 

    std::vector<std::string> keys ; 
    std::vector<double>      vals ; 

    for(unsigned i=0 ; i < msd.size() ; i++)
    { 
        const std::string& key = it->first ; 
        const double&      val = it->second ; 

        keys.push_back(key);
        vals.push_back(val); 

        std::advance(it, 1);  
    }
 
    NP* a = MakeValues( keys, vals ); 
    return a ; 
} 

inline std::string NPX::Desc_MSD(const std::map<std::string, double>& msd) // static
{
    std::stringstream ss ; 
    ss << "NPX::Desc_MSD" << std::endl ; 

    typedef std::map<std::string, double> MSD ; 
    MSD::const_iterator it = msd.begin(); 
    for(unsigned i=0 ; i < msd.size() ; i++)
    { 
        const std::string& key = it->first ; 
        const double& val = it->second ; 
        ss << " key " << key << " val " << val << std::endl ;  
        std::advance(it, 1);  
    }
    std::string s = ss.str(); 
    return s ; 
}

inline NP* NPX::ArrayFromEnumMap( const std::map<int, std::string>& catMap) 
{
    unsigned num_cat = catMap.size() ; 
    NP* a = NP::Make<int>(num_cat); 
    int* aa = a->values<int>() ; 
    typedef std::map<int, std::string> MIS ; 
    MIS::const_iterator it = catMap.begin(); 

    for(unsigned i=0 ; i < num_cat ; i++)
    {
        aa[i] = it->first ; 
        a->names.push_back(it->second) ; 
        std::advance(it, 1);  
    }
    return a ; 
}

inline NP* NPX::MakeCharArray( const std::vector<std::string>& nn )
{
    NP::INT ni = NP::INT(nn.size()); 
    NP::INT maxlen = 0 ; 
    for(NP::INT i=0 ; i < ni ; i++) maxlen = std::max( NP::INT(strlen(nn[i].c_str())), maxlen ) ;
    NP::INT nj = maxlen + 1 ; 

    NP* a = NP::Make<char>(ni, nj) ; 
    char* aa = a->values<char>() ; 
    for(NP::INT i=0 ; i < ni ; i++) for(NP::INT j=0 ; j < nj ; j++) aa[i*nj+j] = nn[i].c_str()[j] ; 
    return a ;  
}

/**
NPX::BOA
---------

Forms the ratio of two columns obtained from two 2D arrays a and b 
with shapes (N,M_a) and (N, M_b) creating an (N, 4) array with 
B/A in third column and A/B in fourth.

T: int or int64_t of the source arrays 
F: float or double of the created array 

**/

template<typename F, typename T>
inline NP* NPX::BOA( NP* a, NP* b, NP::INT a_column, NP::INT b_column, std::ostream* out )  // static
{
    if(out) *out 
       << "NPX::BOA"
       << std::endl 
       << " A " << ( a ? a->sstr() : "-" )
       << std::endl 
       << " B " << ( b ? b->sstr() : "-" )
       << std::endl 
       ;

    bool abort = a == nullptr || b == nullptr ;
    if(abort && out) *out << "NPX::BOA ABORT A or B null \n" ; 
    if(abort) return nullptr ; 

    assert( a->shape.size() == 2 ); 
    assert( b->shape.size() == 2 ); 

    NP::INT a_ni = a->shape[0] ;  
    NP::INT b_ni = b->shape[0] ;  

    if(a->names.size() == 0) for(NP::INT i=0 ; i < a_ni ; i++) a->names.push_back( U::FormName_("A", i, nullptr )) ; 
    if(b->names.size() == 0) for(NP::INT i=0 ; i < b_ni ; i++) b->names.push_back( U::FormName_("B", i, nullptr )) ; 

    assert( NP::INT(a->names.size()) == a_ni ); 
    assert( NP::INT(b->names.size()) == b_ni ); 

    assert( a_ni == b_ni ); 
    NP::INT ni = a_ni ; 

    NP::INT a_nj = a->shape[1] ; 
    NP::INT b_nj = b->shape[1] ; 

    const T* aa = a->cvalues<T>();  
    const T* bb = b->cvalues<T>();  

    NP::INT c_ni = ni ; 
    NP::INT c_nj = 4 ; 

    NP* c = NP::Make<F>(c_ni, c_nj); 
    c->set_meta<std::string>("creator", "NPX::BOA"); 
    c->set_meta<int>("a_column", a_column ); 
    c->set_meta<int>("b_column", b_column ); 

    F* cc = c->values<F>(); 

    c->labels = new std::vector<std::string>(c_nj)  ; 
    (*c->labels)[0] = "A" ; 
    (*c->labels)[1] = "B" ; 
    (*c->labels)[2] = "B/A" ; 
    (*c->labels)[3] = "A/B" ; 

    for(NP::INT i=0 ; i < ni ; i++)
    {
        T av = aa[i*a_nj+a_nj+a_column] ; 
        T bv = bb[i*b_nj+b_nj+b_column] ; 
        F boa = F(bv)/F(av); 
        F aob = F(av)/F(bv); 

        cc[i*c_nj+0] = F(av) ;   
        cc[i*c_nj+1] = F(bv) ;   
        cc[i*c_nj+2] = boa ; 
        cc[i*c_nj+3] = aob ; 

        const char* an = a->names[i].c_str() ; 
        const char* bn = b->names[i].c_str() ; 
        std::string name = U::FormName_(bn,":", an ); 
        c->names.push_back(name) ; 
    }
    return c ; 
}


