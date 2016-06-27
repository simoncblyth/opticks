
#ifdef _MSC_VER

// some posix warning
#define strdup _strdup


// boost::split warnings 
// http://stackoverflow.com/questions/14141476/warning-with-boostsplit-when-compiling
// https://msdn.microsoft.com/en-us/library/aa985974.aspx
// needed include before the boost algorithm string include, not just before the boost::split usage
//
// noted that this define needs to be prior to cstdlib for the warnings to be quelled
// that means prior to a boatload of other headers
//
// done via CMAKE_CXX_FLAGS
//#define _SCL_SECURE_NO_WARNINGS  


// windows defines of min and max prevent std::min std::max from working 
#undef min
#undef max


// object allocated on the heap may not be aligned 16
// CDetector.cc(83): warning C4316: 'NBoundingBox': object allocated on the heap may not be aligned 16
// occurs for any object with glm vec members
// https://github.com/g-truc/glm/issues/235
// apparently fixed by 0.9.7.1 Release : currently on 0.9.6.3

#pragma warning( disable : 4316 )


// warning C4244: '=': conversion from 'int' to 'char'
#pragma warning( disable : 4244 )


//  warning C4189: 'prior': local variable is initialized but not referenced
#pragma warning( disable : 4189 )

//  warning C4459: declaration of 's' hides global declaration 
#pragma warning( disable : 4459 )




#endif

