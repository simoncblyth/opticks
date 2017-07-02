#pragma once

#include <cstdlib>
#include <cstring>
#include <string>
#include <map>
#include <vector>

#include "NSequence.hpp"


#include "NPY_API_EXPORT.hh"
#include "NPY_HEAD.hh"

class NPY_API Index : public NSequence {
   public:
        typedef std::vector<std::string> VS ;
   public:
        Index(const char* itemtype, const char* title=NULL, bool onebased=true);
   public:
        static Index* load(const char* pfold, const char* rfold, const char* itemtype);
        static Index* load(const char* idpath, const char* itemtype);
        static std::string directory(const char* pfold, const char* rfold);
        bool exists(const char* idpath);
        void save(const char* idpath);
        void save(const char* pfold, const char* rfold);
        std::string description();
   public:
       // debugging only
        std::string getPath(const char* idpath, const char* prefix);
        void dumpPaths(const char* idpath, const char* msg="Index::dumpPaths");
   public:
        const char* getItemType();
        const char* getTitle();
        bool isOneBased();     
   public:
        void setTitle(const char* title);
   public:
        int* getSelectedPtr();
        int  getSelected();
        const char* getSelectedKey();
   public:
        // fulfil NSequence, in order to use with GAttrSequence
        unsigned int getNumKeys();
        const char* getKey(unsigned int i);
        unsigned int getIndex(const char* key);
   private:
        void loadMaps(const char* idpath);
        void crossreference();
   public:
        void add(const VS& vs);
        void add(const char* name, unsigned int source, bool sort=true);

        void sortNames(); // currently by ascending local index : ie addition order
        std::vector<std::string>& getNames();
        bool operator() (const std::string& a, const std::string& b);
   public:
        std::string getPrefixedString(const char* tail);
        void setExt(const char* ext);
        unsigned getIndexLocal(const char* name, unsigned missing=0);
        unsigned getIndexSource(const char* name, unsigned missing=0);
        unsigned getIndexSourceStarting(const char* name, unsigned missing=0) ;

        unsigned int getIndexSourceTotal();
        float        getIndexSourceFraction(const char* name);

        bool         hasItem(const char* key);
        const char* getNameLocal(unsigned int local, const char* missing=NULL);
        const char* getNameSource(unsigned int source, const char* missing=NULL);

        unsigned int convertLocalToSource(unsigned int local, unsigned int missing=0);
        unsigned int convertSourceToLocal(unsigned int source, unsigned int missing=0);

   public:
        unsigned int getNumItems();
        void test(const char* msg="Index::test", bool verbose=true);
        void dump(const char* msg="Index::dump");

   private:
        const char*                          m_itemtype ; 
        const char*                          m_title ; 
        const char*                          m_ext ; 
        int                                  m_selected ; 
        bool                                 m_onebased ; 
        unsigned int                         m_source_total ; 
        std::map<std::string, unsigned int>  m_source ; 
        std::map<std::string, unsigned int>  m_local ; 
        std::map<unsigned int, unsigned int> m_source2local ; 
        std::map<unsigned int, unsigned int> m_local2source ; 
        std::vector<std::string>             m_names ; 
   private:
        // populated by formTable
        std::vector<std::string>             m_labels ; 
        std::vector<unsigned int>            m_codes ; 
};

#include "NPY_TAIL.hh"

