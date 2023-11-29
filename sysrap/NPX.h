#pragma once
/**
NPX.h : NP.hh related extras such as static converters
=========================================================


**/

#include "NP.hh"


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


    static NP* MakeDemo(const char* dtype="<f4" , int ni=-1, int nj=-1, int nk=-1, int nl=-1, int nm=-1, int no=-1 ); 

    template<typename T> static NP*  Make( const std::vector<T>& src ); 
    template<typename T> static NP*  Make( T d0, T v0, T d1, T v1 ); 
    template<typename T, typename... Args> static NP*  Make(const T* src, Args ... shape );  // TODO rename ArrayFromData

    template<typename T> static NP* FromString(const char* str, char delim=' ') ;  


    static NP* Holder( const std::vector<std::string>& names ); 



    template<typename T, typename S> 
    static NP* ArrayFromVec__(const std::vector<S>& v, const std::vector<int>& itemshape ); 

    template<typename T, typename S, typename... Args> 
    static NP* ArrayFromVec(const std::vector<S>& v, Args ... args_itemshape );   // ArrayFromVec_ellipsis

    template<typename T, typename S> 
    static NP* ArrayFromVec_(const std::vector<S>& v, const char* str_itemshape );  


    template<typename S> 
    static void VecFromArray(std::vector<S>& v, const NP* a );

    template<typename S>
    static int VecFromMap( std::vector<S>& v,  const std::map<int, S>& m, bool contiguous_key=true ); 
    template<typename S>
    static void MapFromVec( std::map<int, S>& m,  const std::vector<S>& v, int k0=0, bool contiguous_key=true ); 

    template<typename T, typename S>
    static NP* ArrayFromMap( const std::map<int, S>& m, bool contiguous_key=true ); 
    template<typename S>
    static void MapFromArray( std::map<int, S>& m, const NP* a ); 

    template<typename S>
    static NP* ArrayFromDiscoMap( const std::map<int, S>& m ); 
    template<typename S>
    static void DiscoMapFromArray( std::map<int, S>& m, const NP* a ); 
    template<typename S>
    static std::string DescDiscoMap( const std::map<int, S>& m ); 

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
    static NP* BOA( NP* a, NP* b, int a_column=-1, int b_column=-1, std::ostream* out=nullptr  ); 

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

    for(unsigned i=0 ; i < values.size() ; i++)
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


inline NP* NPX::MakeDemo(const char* dtype, int ni, int nj, int nk, int nl, int nm, int no )
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
inline NP* NPX::Make(const T* src, Args ... args )   // TODO rename ArrayFromData
{
    std::string dtype = descr_<T>::dtype() ; 
    std::vector<int> shape = {args...};
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










template<typename T, typename S> 
inline NP* NPX::ArrayFromVec__(const std::vector<S>& v, const std::vector<int>& itemshape )   
{
    assert( sizeof(S) >= sizeof(T) );  
    int ni = v.size() ; 
    int nj = sizeof(S) / sizeof(T) ; 

    const T* src = (T*)v.data() ; 

    std::vector<int> shape ; 
    shape.push_back(ni) ; 

    if(itemshape.size() == 0 )
    {
        shape.push_back(nj) ; 
    }
    else 
    {
        int itemcheck = 1 ; 
        for(unsigned i=0 ; i < itemshape.size() ; i++)  
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
    std::vector<int> itemshape = {args_itemshape...};
    return ArrayFromVec__<T,S>(v, itemshape ); 
}

template<typename T, typename S> 
inline NP* NPX::ArrayFromVec_(const std::vector<S>& v, const char* str_itemshape )
{
    std::vector<int> itemshape ; 
    U::MakeVec(itemshape,  str_itemshape, ',' ); 
    return ArrayFromVec__<T,S>(v, itemshape ); 
}




template<typename S> 
inline void NPX::VecFromArray(std::vector<S>& v, const NP* a )
{
   if(a == nullptr || a->shape.size() == 0 ) return ; 
   int ni = a->shape[0] ; 
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

**/


template<typename S>
inline int NPX::VecFromMap( std::vector<S>& v,  const std::map<int, S>& m, bool contiguous_key ) // static
{
    int ni = int(m.size()) ; 

    v.clear(); 
    v.resize(ni); 

    typename std::map<int,S>::const_iterator it = m.begin()  ;
    int k0 = it->first ; 

    for(int idx=0 ; idx < ni ; idx++)
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
NPX::MapFromVec
----------------

HMM: to support contiguous_key:false would need type S to follow 
some convention such as keeping int keys within the first 32 bit member. 

**/

template<typename S>
inline void NPX::MapFromVec( std::map<int, S>& m,  const std::vector<S>& v, int k0, bool contiguous_key )
{
    assert( contiguous_key == true ); 

    int ni = int(v.size()) ; 
    m.clear(); 

    for(int i=0 ; i < ni ; i++)
    {
        const S& item = v[i] ; 
        int key = k0 + i ;  
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
template<>
inline NP* NPX::ArrayFromDiscoMap( const std::map<int,int>& m )
{
    int ni = m.size() ; 
    int nj = 2 ; 
    NP* a = NP::Make<int>(ni, nj) ;  
    int* aa = a->values<int>(); 

    typedef std::map<int,int> MII ; 
    MII::const_iterator it = m.begin(); 

    for(int i=0 ; i < ni ; i++)
    {
        aa[i*nj+0] = it->first ;  
        aa[i*nj+1] = it->second ;  
        it++ ; 
    }
    return a ; 
}


template<typename S>
inline void NPX::DiscoMapFromArray( std::map<int, S>& m, const NP* a ){}

template<>
inline void NPX::DiscoMapFromArray( std::map<int, int>& m, const NP* a )
{
    assert( a && a->uifc == 'i' && a->ebyte == 4 && a->shape.size() == 2 ); 
    int ni = a->shape[0] ; 
    int nj = a->shape[1] ;
    assert( nj == 2 );  

    const int* aa = a->cvalues<int>(); 
    for(int i=0 ; i < ni ; i++)
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

template<>
inline std::string NPX::DescDiscoMap( const std::map<int,int>& m )
{
    int ni = m.size() ; 
    typedef std::map<int,int> MII ; 
    MII::const_iterator it = m.begin(); 
    std::stringstream ss ; 
    ss << "NPX::DescDiscoMap" << std::endl << " m.size " << ni << std::endl ; 
    for(int i=0 ; i < ni ; i++)
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
        for(int j=0 ; j < int(fields.size()) ; j++ )
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
    assert( int(a->names.size()) == a->shape[0] );

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
    int ni = int(nn.size()); 
    int maxlen = 0 ; 
    for(int i=0 ; i < ni ; i++) maxlen = std::max( int(strlen(nn[i].c_str())), maxlen ) ;
    int nj = maxlen + 1 ; 

    NP* a = NP::Make<char>(ni, nj) ; 
    char* aa = a->values<char>() ; 
    for(int i=0 ; i < ni ; i++) for(int j=0 ; j < nj ; j++) aa[i*nj+j] = nn[i].c_str()[j] ; 
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
inline NP* NPX::BOA( NP* a, NP* b, int a_column, int b_column, std::ostream* out )  // static
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
    if(abort) std::cerr 
        << "NPX::BOA ABORT A or B null "
        << std::endl 
        << " A " << ( a ? a->sstr() : "-" )
        << std::endl 
        << " B " << ( b ? b->sstr() : "-" )
        << std::endl 
        ;
    if(abort) return nullptr ; 

    assert( a->shape.size() == 2 ); 
    assert( b->shape.size() == 2 ); 

    int a_ni = a->shape[0] ;  
    int b_ni = b->shape[0] ;  

    if(a->names.size() == 0) for(int i=0 ; i < a_ni ; i++) a->names.push_back( U::FormName_("A", i, nullptr )) ; 
    if(b->names.size() == 0) for(int i=0 ; i < b_ni ; i++) b->names.push_back( U::FormName_("B", i, nullptr )) ; 

    assert( int(a->names.size()) == a_ni ); 
    assert( int(b->names.size()) == b_ni ); 

    assert( a_ni == b_ni ); 
    int ni = a_ni ; 

    int a_nj = a->shape[1] ; 
    int b_nj = b->shape[1] ; 

    const T* aa = a->cvalues<T>();  
    const T* bb = b->cvalues<T>();  

    int c_ni = ni ; 
    int c_nj = 4 ; 

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

    for(int i=0 ; i < ni ; i++)
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


