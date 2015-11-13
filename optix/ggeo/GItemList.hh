#pragma once

#include <string>
#include <vector>
#include <map>

#include "NSequence.hpp"

class GItemList : public NSequence {
   public:
       static unsigned int UNSET ; 
       static const char* GITEMLIST ; 
       static GItemList* load(const char* idpath, const char* itemtype, const char* reldir=NULL);
   public:
       GItemList(const char* itemtype, const char* reldir=NULL);
       void add(const char* name);
       void save(const char* idpath);
       void dump(const char* msg="GItemList::dump");
    public:
       void dumpFields(const char* msg="GItemList::dumpFields", const char* delim="/", unsigned int fwid=30);
       void replaceField(unsigned int field, const char* from, const char* to, const char* delim="/");
    public:
       // fulfil NSequence protocol
       unsigned int getNumKeys();
       void setKey(unsigned int index, const char* newkey);
       const char* getKey(unsigned int index);
       unsigned int getIndex(const char* key);    // 0-based index of first matching name, OR INT_MAX if no match
   public:
       bool operator()(const std::string& a_, const std::string& b_);
       void setOrder(std::map<std::string, unsigned int>& order);
       void sort();

   private:
       void read_(const char* txtpath);
       void load_(const char* idpath);
   private:
       std::string              m_itemtype ;
       std::string              m_reldir ;
       std::vector<std::string> m_list ; 
       std::map<std::string, unsigned int> m_order ; 
       std::string              m_empty ; 
};

inline GItemList::GItemList(const char* itemtype, const char* reldir) : NSequence()
{
    m_itemtype = itemtype ; 
    m_reldir   = reldir ? reldir : GITEMLIST ; 
}

inline void GItemList::add(const char* name)
{
    m_list.push_back(name);
}

inline unsigned int GItemList::getNumKeys()
{
    return m_list.size();
}
inline const char* GItemList::getKey(unsigned int index)
{
    return index < m_list.size() ? m_list[index].c_str() : NULL  ;
}

inline void GItemList::setKey(unsigned int index, const char* newkey)
{
    if(index < m_list.size()) m_list[index] = newkey  ; 
}


inline void GItemList::setOrder(std::map<std::string, unsigned int>& order)
{
    m_order = order ; 
}






