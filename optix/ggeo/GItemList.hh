#pragma once

#include <string>
#include <vector>


class GItemList {
   public:
       static const char* GITEMLIST ; 
       static GItemList* load(const char* idpath, const char* itemtype);
   public:
       GItemList(const char* itemtype);
       void add(const char* name);
       void save(const char* idpath);
       void dump(const char* msg="GItemList::dump");
   private:
       void read_(const char* txtpath);
       void load_(const char* idpath);
   private:
       std::string              m_itemtype ;
       std::vector<std::string> m_list ; 
};

inline GItemList::GItemList(const char* itemtype)
{
    m_itemtype = itemtype ; 
}

inline void GItemList::add(const char* name)
{
    m_list.push_back(name);
}




