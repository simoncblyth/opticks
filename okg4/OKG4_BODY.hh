
#ifdef _MSC_VER

#define strdup _strdup

// object allocated on the heap may not be aligned 16
// https://github.com/g-truc/glm/issues/235
// apparently fixed by 0.9.7.1 Release : currently on 0.9.6.3

#pragma warning( disable : 4316 )

#endif

