#pragma once
#include <map>
#include <vector>
#include <string>

//#include "BRAP_API_EXPORT.hh"

class BJson {
     public:


    template<typename A, typename B> 
    static void saveMap( typename std::map<A,B> & mp, const char* dir, const char* name) ;

    template<typename A, typename B>
    static void saveMap( typename std::map<A,B>& mp, const char* path) ;



    template<typename A, typename B> 
    static int loadMap( typename std::map<A,B> & mp, const char* dir, const char* name, unsigned int depth=0) ;

    template<typename A, typename B>
    static int loadMap( typename std::map<A,B>& mp, const char* path, unsigned int depth=0) ;



    template<typename A, typename B>
    static void dumpMap( typename std::map<A,B>& mp, const char* msg="jsonutil.dumpMap " ) ;





    template<typename A, typename B> 
    static void saveList( typename std::vector<std::pair<A,B> > & vp, const char* dir, const char* name);

    template<typename A, typename B> 
    static void saveList( typename std::vector<std::pair<A,B> > & vp, const char* path);



    template<typename A, typename B> 
    static void loadList( typename std::vector<std::pair<A,B> > & vp, const char* dir, const char* name);

    template<typename A, typename B> 
    static void loadList( typename std::vector<std::pair<A,B> > & vp, const char* path);



    template<typename A, typename B> 
    static void dumpList( typename std::vector<std::pair<A,B> > & vp, const char* msg="jsonutil.dumpList");


};





