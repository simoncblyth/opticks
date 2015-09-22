#include "GItemList.hh"

#include <iostream>
#include <fstream>
#include <ostream>   
#include <algorithm>
#include <iterator>

#include <boost/log/trivial.hpp>
#define LOG BOOST_LOG_TRIVIAL
// trace/debug/info/warning/error/fatal

#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;

const char* GItemList::GITEMLIST = "GItemList" ; 

GItemList* GItemList::load(const char* idpath, const char* itemtype)
{
    GItemList* gil = new GItemList(itemtype);
    gil->load_(idpath); 
    return gil ;
}

void GItemList::load_(const char* idpath)
{
   fs::path cachedir(idpath);
   fs::path typedir(cachedir / GITEMLIST );

   if(fs::exists(typedir) && fs::is_directory(typedir))
   {
       fs::path txtpath(typedir);
       txtpath /= m_itemtype + ".txt" ; 

       if(fs::exists(txtpath) && fs::is_regular_file(txtpath))     
       {
            read_(txtpath.string().c_str());
       }
   }
}

void GItemList::read_(const char* txtpath)
{
   LOG(info) << "GItemList::read_ " << txtpath ;  

   std::ifstream ifs(txtpath);

   std::copy(std::istream_iterator<std::string>(ifs), 
             std::istream_iterator<std::string>(),
             std::back_inserter(m_list)); 

   ifs.close();
}


void GItemList::save(const char* idpath)
{
   fs::path cachedir(idpath);
   fs::path typedir(cachedir / GITEMLIST );

   if(!fs::exists(typedir))
   {
        if (fs::create_directories(typedir))
        {
            printf("GItemList::save created directory %s \n", typedir.string().c_str() );
        }
   }

   if(fs::exists(typedir) && fs::is_directory(typedir))
   {
       fs::path txtpath(typedir);
       txtpath /= m_itemtype + ".txt" ; 

       LOG(info) << "GItemList::save writing to " << txtpath.string() ;       

       std::ofstream ofs(txtpath.string());
       std::copy(m_list.begin(),m_list.end(),std::ostream_iterator<std::string>( ofs,"\n"));
       ofs.close();
   }
}

void GItemList::dump(const char* msg)
{
    LOG(info) << msg ; 
    std::copy( m_list.begin(),m_list.end(), std::ostream_iterator<std::string>(std::cout, "\n"));
}

