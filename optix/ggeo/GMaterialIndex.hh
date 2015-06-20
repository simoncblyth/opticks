#pragma once

#include <string>
#include <map>

class GItemIndex {
   public:
        static const char* SOURCE_NAME ; 
        static const char* LOCAL_NAME  ; 
        static GItemIndex* load(const char* idpath);
   public:
        GItemIndex();
        void save(const char* idpath);

   public:
        unsigned int getIndexLocal(const char* name, unsigned int missing=0);
        unsigned int getIndexSource(const char* name, unsigned int missing=0);

        const char* getNameLocal(unsigned int local, const char* missing=NULL);
        const char* getNameSource(unsigned int source, const char* missing=NULL);

        unsigned int convertLocalToSource(unsigned int local, unsigned int missing=0);
        unsigned int convertSourceToLocal(unsigned int source, unsigned int missing=0);

   private:
        void loadMaps(const char* idpath);
        void crossreference();
   public:
        // invoked from GBoundaryLib::createWavelengthAndOpticalBuffers
        void add(const char* name, unsigned int index);
   public:
        unsigned int getNumMaterials();
        bool operator() (const std::string& a, const std::string& b);
        void dump(const char* msg="GItemIndex::dump");
        void test(const char* msg="GItemIndex::test");

   private:
        std::map<std::string, unsigned int>  m_source ; 
        std::map<std::string, unsigned int>  m_local ; 
        std::map<unsigned int, unsigned int> m_source2local ; 
        std::map<unsigned int, unsigned int> m_local2source ; 
};

inline GItemIndex::GItemIndex()
{
}

