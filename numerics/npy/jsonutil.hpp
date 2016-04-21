#include <map>
#include <vector>
#include <string>

bool existsPath(const char* path );
bool existsPath(const char* dir, const char* name );
std::string preparePath(const char* dir_, const char* name, bool create=false );



template<typename A, typename B> 
void saveMap( typename std::map<A,B> & mp, const char* dir, const char* name) ;

template<typename A, typename B>
void saveMap( typename std::map<A,B>& mp, const char* path) ;



template<typename A, typename B> 
int loadMap( typename std::map<A,B> & mp, const char* dir, const char* name, unsigned int depth=0) ;

template<typename A, typename B>
int loadMap( typename std::map<A,B>& mp, const char* path, unsigned int depth=0) ;




template<typename A, typename B> 
void saveList( typename std::vector<std::pair<A,B> > & vp, const char* dir, const char* name);

template<typename A, typename B> 
void saveList( typename std::vector<std::pair<A,B> > & vp, const char* path);




template<typename A, typename B> 
void loadList( typename std::vector<std::pair<A,B> > & vp, const char* dir, const char* name);

template<typename A, typename B> 
void loadList( typename std::vector<std::pair<A,B> > & vp, const char* path);




template<typename A, typename B>
void dumpMap( typename std::map<A,B>& mp, const char* msg="jsonutil.dumpMap " ) ;

template<typename A, typename B> 
void dumpList( typename std::vector<std::pair<A,B> > & vp, const char* msg="jsonutil.dumpList");





