#pragma once

#include <string>
#include <vector>
#include <map>

struct NSlice ;
#include "NSequence.hpp"

// TODO: rename to GNameList/GKeyList/GStringList  

#include "GGEO_API_EXPORT.hh"
#include "GGEO_HEAD.hh"

class GGEO_API GItemList : public NSequence {
   public:
       static unsigned int UNSET ; 
       static const char* GITEMLIST ; 
       static GItemList* load(const char* idpath, const char* itemtype, const char* reldir=NULL);
       static GItemList* Repeat( const char* itemtype, const char* name, unsigned numRepeats, const char* reldir=NULL );
   public:
       GItemList(const char* itemtype, const char* reldir=NULL);
       void add(const char* name);
       void add(GItemList* other);
       void save(const char* idpath);
       void save(const char* idpath, const char* reldir, const char* txtname);  // for debug 
       void dump(const char* msg="GItemList::dump");
       const std::string& getRelDir() const ;  
    public:
       GItemList* make_slice(const char* slice);
       GItemList* make_slice(NSlice* slice);
    public:
       void dumpFields(const char* msg="GItemList::dumpFields", const char* delim="/", unsigned int fwid=30);
       void replaceField(unsigned int field, const char* from, const char* to, const char* delim="/");
    public:
       unsigned int getNumItems();
    public:
       // fulfil NSequence protocol
       const char* getKey(unsigned index) const ;
       unsigned int getNumKeys() const ;
       unsigned int getIndex(const char* key) const ;    // 0-based index of first matching name, OR UINT_MAX if no match
    public:
       void setKey(unsigned int index, const char* newkey);
       static bool isUnset(unsigned int index);
   public:
       void getIndicesWithKeyEnding( std::vector<unsigned>& indices, const char* ending ) const ;  
   public:
       bool operator()(const std::string& a_, const std::string& b_);
       void setOrder(std::map<std::string, unsigned int>& order);
       void sort();
   public:
       void getCurrentOrder( std::map<std::string, unsigned int>& order );
   private:
       void save_(const char* txtpath);
       void read_(const char* txtpath);
       void load_(const char* idpath);
   private:
       std::string              m_itemtype ;
       std::string              m_reldir ;
       std::vector<std::string> m_list ; 
       std::map<std::string, unsigned int> m_order ; 
       std::string              m_empty ; 
};

#include "GGEO_TAIL.hh"

