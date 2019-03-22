#ifdef __clang__
#pragma clang diagnostic push
// gleq.h:75:9: warning: anonymous types declared in an anonymous union are an extension
#pragma clang diagnostic ignored "-Wnested-anon-types"
#endif


#define NEWGLEQ 1

#ifdef NEWGLEQ
#include "gleq.h"
#else
#include "old_gleq.h"
#endif

#ifdef __clang__
#pragma clang diagnostic pop
#endif



