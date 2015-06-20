#include <map>
#include <string>


template<typename A, typename B> 
void saveMap( typename std::map<A,B> & mp, const char* dir, const char* name) ;

template<typename A, typename B>
void saveMap( typename std::map<A,B>& mp, const char* path) ;



template<typename A, typename B> 
void loadMap( typename std::map<A,B> & mp, const char* dir, const char* name) ;

template<typename A, typename B>
void loadMap( typename std::map<A,B>& mp, const char* path) ;



template<typename A, typename B>
void dumpMap( typename std::map<A,B>& mp, const char* msg="jsonutil.dumpMap " ) ;


