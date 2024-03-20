#include "NP.hh"
#include "sprof.h"


int main(int argc, char** argv)
{
    const char* source = U::Resolve("$SPROF_PATH") ; 
    const char* ptn = U::GetEnv("SPROF_PTN", "QSim__")  ; 
    const char* SPROF_DUMMY_LAST = U::GetEnv("SPROF_DUMMY_LAST", nullptr ); 

    const char* prof = source ? U::ReadString(source) : nullptr ; 
    if(prof == nullptr) return 1 ; 

    std::string _dest = U::ChangeExt( source, ".txt", "_txt.npy" );  // not .npy to avoid stompage
    const char* dest = _dest.c_str(); 
    //std::cout << "prof:" << ( prof ? prof : "-" ) << std::endl ; 


    std::vector<std::string> keys ; 
    std::vector<std::string> vals ; 
    bool only_with_profile = true ; 

    NP::GetMetaKV_(prof, &keys, &vals, only_with_profile, ptn ); 
    assert( keys.size() == vals.size() ); 
    int num_keys0 = keys.size() ; 
    int num_keys = SPROF_DUMMY_LAST ? num_keys0 + 1 : num_keys0 ; 

    std::cout 
         << " num_keys0 " << num_keys0 << std::endl 
         << " num_keys " << num_keys << std::endl 
         << "  SPROF_DUMMY_LAST " << ( SPROF_DUMMY_LAST ? SPROF_DUMMY_LAST : "-" ) << std::endl
         ; 

    NP* a = NP::Make<int64_t>( num_keys, 3 ); 
    int64_t* aa = a->values<int64_t>(); 

    a->set_meta<std::string>("creator", "sysrap/tests/sprof.cc"); 
    a->set_meta<std::string>("source", source ); 
    a->set_meta<std::string>("dest", dest ); 
    a->set_meta<std::string>("ptn", ptn ); 
    a->set_meta<std::string>("SPROF_DUMMY_LAST", ( SPROF_DUMMY_LAST ? SPROF_DUMMY_LAST : "-" ) ); 

    int edge = 10 ; 

    for(int i=0 ; i < num_keys0 ; i++)
    {
         const char* k = keys[i].c_str(); 
         const char* v = vals[i].c_str(); 

         sprof p = {} ; 
         sprof::Import(p, v ); 

         if( i < edge || i >= num_keys - edge   ) std::cout 
              << std::setw(10) << i << " : " 
              << std::setw(20) << k << " : " 
              << std::setw(30) << v << " : " 
              << sprof::Desc_(p) 
              << std::endl
              ; 

         aa[3*i+0] = p.st ; 
         aa[3*i+1] = p.vm ; 
         aa[3*i+2] = p.rs ; 

         a->names.push_back(k) ; 

         if( SPROF_DUMMY_LAST && i == num_keys0 - 1 )
         {
             aa[3*(i+1)+0] = p.st ;  // dumplicate prior as dummy last  
             aa[3*(i+1)+1] = p.vm ; 
             aa[3*(i+1)+2] = p.rs ; 
             a->names.push_back(SPROF_DUMMY_LAST) ; 
         } 
    }


    std::cout << "saving to dest [" << ( dest ? dest : "-" ) << std::endl ;  
    a->save(dest); 

    return 0 ; 
}
