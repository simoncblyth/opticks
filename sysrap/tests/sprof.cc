#include "NP.hh"
#include "sprof.h"


int main(int argc, char** argv)
{
    const char* source = U::Resolve("$SPROF_PATH") ; 
    const char* ptn = U::GetEnv("SPROF_PTN", "QSim__")  ; 

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
    int num_keys = keys.size() ; 

    std::cout << " num_keys " << num_keys << std::endl ; 

    NP* a = NP::Make<int64_t>( num_keys, 3 ); 
    int64_t* aa = a->values<int64_t>(); 

    a->set_meta<std::string>("creator", "sysrap/tests/sprof.cc"); 
    a->set_meta<std::string>("source", source ); 
    a->set_meta<std::string>("dest", dest ); 
    a->set_meta<std::string>("ptn", ptn ); 

    int edge = 10 ; 

    for(int i=0 ; i < num_keys ; i++)
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
    }

    std::cout << "saving to dest [" << ( dest ? dest : "-" ) << std::endl ;  
    a->save(dest); 

    return 0 ; 
}
