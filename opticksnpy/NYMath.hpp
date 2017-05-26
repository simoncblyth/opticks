#pragma once


// https://useyourloaf.com/blog/disabling-clang-compiler-warnings/

#ifdef __clang__

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-braces"
#pragma clang diagnostic ignored "-Wgnu-anonymous-struct"
#pragma clang diagnostic ignored "-Wnested-anon-types"
#pragma clang diagnostic ignored "-Wunused-parameter"


#elif defined(__GNUC__) || defined(__GNUG__)
#elif defined(_MSC_VER)
#endif


#include "YoctoGL/yocto_math.h"



#ifdef __clang__

#pragma clang diagnostic pop

#elif defined(__GNUC__) || defined(__GNUG__)
#elif defined(_MSC_VER)
#endif





