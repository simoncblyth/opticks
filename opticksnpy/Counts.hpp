#pragma once
#include <string>
#include <vector>
#include <map>

class Index ; 


#include "NPY_API_EXPORT.hh"
#include "NPY_HEAD.hh"

template<typename T>
class NPY_API Counts {
    public:
          typedef typename std::map<std::string, T>    MST ; 
          typedef typename std::pair<std::string, T>    ST ; 
          typedef typename std::vector<ST>             VST ; 
          typedef typename VST::iterator              VSTI ; 
    public: 
          // persistency 
          static Counts<T>* load(const char* path);
          static Counts<T>* load(const char* dir, const char* name);
          void save(const char* path);
          void save(const char* dir, const char* name);
    public:
          Index* make_index(const char* itemtype, const char* reldir);
    public:
          Counts(const char* name="Counts");
    public:
          void add(const char* key, T count=1);
          void addPair(const ST& p);
          void addMap( const MST& m);
    public:
          void checkfind(const char* key);
          typename std::vector<std::pair<std::string, T> >::iterator find(const char* key);
    public:
          unsigned int size();
          std::pair<std::string, T>& get(unsigned int index); 
          T                          getCount(const char* key); 
    public:
          void sort(bool ascending=true);
          void dump(const char* msg="Counts::dump", unsigned long nline=32);
    public:
          std::vector<ST>& counts();          
    private:
          static bool ascending_count_order(const ST& a, const ST& b);
          static bool descending_count_order(const ST& a, const ST& b);
          void load_(const char* path);
          void load_(const char* dir, const char* name);
    private:
          const char*      m_name ;  
          std::vector<ST>  m_counts  ;
};

#include "NPY_TAIL.hh"


