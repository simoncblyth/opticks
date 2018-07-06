#pragma once

#ifndef USESYMBOL_API

//#define USESYMBOL_API
//#define USESYMBOL_API extern 
#define USESYMBOL_API  __attribute__ ((visibility ("default")))

#endif

namespace UseSymbol 
{
    USESYMBOL_API void Check(); 
}

struct USESYMBOL_API UseSymbolStruct
{
    static void Check(); 
};




