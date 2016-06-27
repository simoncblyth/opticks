
#ifdef _MSC_VER
#pragma warning(push)
// warning C4244: 'argument': conversion from 'double' to 'float', possible loss of data
// from CLHEP/Random headers
#pragma warning( disable : 4244 )

// boost::split warnings 
// http://stackoverflow.com/questions/14141476/warning-with-boostsplit-when-compiling
// https://msdn.microsoft.com/en-us/library/aa985974.aspx
// needed include before the boost algorithm string include, not just before the boost::split usage
#define _SCL_SECURE_NO_WARNINGS  


#endif


