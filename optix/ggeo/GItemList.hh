#pragma once

#include <string>
#include <vector>


class GItemList {
   public:
       static unsigned int UNSET ; 
       static const char* GITEMLIST ; 
       static GItemList* load(const char* idpath, const char* itemtype);
   public:
       GItemList(const char* itemtype);
       void add(const char* name);
       void save(const char* idpath);
       void dump(const char* msg="GItemList::dump");

       unsigned int getNumItems();
       std::string& getItem(unsigned int index);
       unsigned int getIndex(const char* name);  // 0-based index of first matching name, OR INT_MAX if no match
   private:
       void read_(const char* txtpath);
       void load_(const char* idpath);
   private:
       std::string              m_itemtype ;
       std::vector<std::string> m_list ; 
       std::string              m_empty ; 
};

inline GItemList::GItemList(const char* itemtype)
{
    m_itemtype = itemtype ; 
}

inline void GItemList::add(const char* name)
{
    m_list.push_back(name);
}

inline unsigned int GItemList::getNumItems()
{
    return m_list.size();
}

inline std::string& GItemList::getItem(unsigned int index)
{
    return index < m_list.size() ? m_list[index] : m_empty  ;
}


