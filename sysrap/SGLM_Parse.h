#pragma once

/**
SGLM_Parse : minimal command string parsing into key,val,opt strings
-----------------------------------------------------------------------

Old opticks used boost-program-options for liveline parsing, thats a no-no
as dont need to be all that general, and dont need so much flexibility
and dont have too many keys. Can simply use a big if to route config 
to the right place::

    --near 777 --far 7777 --tmin 0.1 --tmax 100 --eye 0,10,0.1 --look 0,0,0

Also can require tokens to always be preceded by "--" "--token value" 
or "--switch --another_switch"

1. split the cmd into elements delimited by spaces
2. iterate thru the elements in pairs collecting 
   into key,val,opt vectors 

**/

#include "sstr.h"

struct SGLM_Parse
{
    std::vector<std::string> key ; 
    std::vector<std::string> val ; 
    std::vector<std::string> opt ; 

    static bool IsKey(const char* str); 
    SGLM_Parse(const char* cmd);
    std::string desc() const ; 
};

inline bool SGLM_Parse::IsKey(const char* str) // static
{
    return str && strlen(str) > 2 && str[0] == '-' && str[1] == '-' ;  
}

inline SGLM_Parse::SGLM_Parse(const char* cmd)
{    
    //std::cout << "SGLM_Parse::SGLM_Parse [" << ( cmd ? cmd : "-" ) << "]" << std::endl; 
    std::vector<std::string> elem ; 
    sstr::Split(cmd,' ',elem); 
    int num_elem = elem.size(); 

    for(int i=0 ; i < std::max(1, num_elem - 1) ; i++)
    {
        const char* e0 = i   < num_elem ? elem[i].c_str()   : nullptr ; 
        const char* e1 = i+1 < num_elem ? elem[i+1].c_str() : nullptr ; 
        bool k0 = IsKey(e0); 
        bool k1 = IsKey(e1); 

        if( ( k0 && k1 ) || (k0 && e1 == nullptr)  )   // "--red --blue" OR "--green" 
        {
            opt.push_back(e0+2); 
        }
        else if(k0 && !k1)  // eg: --eye 0,10,0.1 
        {
            key.push_back(e0+2); 
            val.push_back(e1); 
        } 
    }
}

inline std::string SGLM_Parse::desc() const 
{
    std::stringstream ss ; 
    ss << "SGLM_Parse::desc"
       << " key.size " << key.size()
       << " val.size " << val.size()
       << " opt.size " << opt.size()
       << std::endl 
       ;
    std::string str = ss.str(); 
    return str ; 
}


