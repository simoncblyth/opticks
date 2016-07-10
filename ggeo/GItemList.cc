#include "GItemList.hh"

#include <climits>
#include <iostream>
#include <fstream>
#include <ostream>   
#include <algorithm>
#include <iterator>
#include <iomanip>


#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/join.hpp>
#include <boost/filesystem.hpp>


#include "BFile.hh"
#include "NSlice.hpp"

#include "PLOG.hh"

namespace fs = boost::filesystem;

const char* GItemList::GITEMLIST = "GItemList" ; 

unsigned int GItemList::UNSET = UINT_MAX ; 


bool GItemList::isUnset(unsigned int index)
{
    return index == UNSET ; 
}

unsigned int GItemList::getIndex(const char* key)
{
    if(key)
    {
        for(unsigned int i=0 ; i < m_list.size() ; i++) if(m_list[i].compare(key) == 0) return i ;  
    } 
    return UNSET  ; 
}


GItemList* GItemList::load(const char* idpath, const char* itemtype, const char* reldir)
{
    GItemList* gil = new GItemList(itemtype, reldir);
    gil->load_(idpath); 
    return gil ;
}

void GItemList::load_(const char* idpath)
{

   // TODO: adopt BFile

   /*

   fs::path cachedir(idpath);
   fs::path typedir(cachedir / m_reldir );

   if(fs::exists(typedir) && fs::is_directory(typedir))
   {
       fs::path txtpath(typedir);
       txtpath /= m_itemtype + ".txt" ; 

       if(fs::exists(txtpath) && fs::is_regular_file(txtpath))     
       {
            read_(txtpath.string().c_str());
       }
   }

   */

   std::string txtname = m_itemtype + ".txt" ;  
   std::string txtpath = BFile::FormPath(idpath, m_reldir.c_str(), txtname.c_str()) ;

   if(BFile::ExistsFile(txtpath.c_str()))
   {
       read_(txtpath.c_str());  
   } 
   else
   {
       LOG(warning) << "GItemList::load_"
                    << " NO SUCH TXTPATH " 
                    << txtpath 
                    ;
  
   }


}

void GItemList::read_(const char* txtpath)
{
   LOG(debug) << "GItemList::read_ " << txtpath ;  

   std::ifstream ifs(txtpath);

   std::copy(std::istream_iterator<std::string>(ifs), 
             std::istream_iterator<std::string>(),
             std::back_inserter(m_list)); 

   ifs.close();
}




void GItemList::save(const char* idpath)
{
    std::string txtname = m_itemtype + ".txt" ; 
    std::string txtpath = BFile::preparePath(idpath, m_reldir.c_str(), txtname.c_str() ); 
    LOG(info) << "GItemList::save writing to " << txtpath ;       
    save_(txtpath.c_str());
}

void GItemList::save_(const char* txtpath)
{
    std::ofstream ofs(txtpath);
    std::copy(m_list.begin(),m_list.end(),std::ostream_iterator<std::string>( ofs,"\n"));
    ofs.close();
}


/*
void GItemList::save(const char* idpath)
{
   fs::path cachedir(idpath);
   fs::path typedir(cachedir / m_reldir );

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

       std::string path = txtpath.string();
       LOG(info) << "GItemList::save writing to " << path ;       
       save_(path.c_str());

   }
}
*/





GItemList::GItemList(const char* itemtype, const char* reldir) : NSequence()
{
    m_itemtype = itemtype ; 
    m_reldir   = reldir ? reldir : GITEMLIST ; 
}

void GItemList::add(const char* name)
{
    m_list.push_back(name);
}

unsigned int GItemList::getNumItems()
{
    return m_list.size();
}
unsigned int GItemList::getNumKeys()
{
    return m_list.size();
}

const char* GItemList::getKey(unsigned int index)
{
    return index < m_list.size() ? m_list[index].c_str() : NULL  ;
}

void GItemList::setKey(unsigned int index, const char* newkey)
{
    if(index < m_list.size()) m_list[index] = newkey  ; 
}


void GItemList::setOrder(std::map<std::string, unsigned int>& order)
{
    m_order = order ; 
}





void GItemList::dump(const char* msg)
{
    LOG(info) << msg ; 
    //std::copy( m_list.begin(),m_list.end(), std::ostream_iterator<std::string>(std::cout, "\n"));

    for(unsigned int i=0 ; i < m_list.size() ; i++)
         std::cout << std::setw(4) << i 
                   << " : "
                   << m_list[i]
                   << std::endl ;
}


bool GItemList::operator()(const std::string& a_, const std::string& b_)
{
    std::map<std::string, unsigned int>::const_iterator end = m_order.end() ; 
    unsigned int ia = m_order.find(a_) == end ? UINT_MAX :  m_order[a_] ; 
    unsigned int ib = m_order.find(b_) == end ? UINT_MAX :  m_order[b_] ; 
    return ia < ib ; 
}

void GItemList::sort()
{
    if(m_order.size() == 0) return ; 
    std::stable_sort( m_list.begin(), m_list.end(), *this );
}


void GItemList::dumpFields(const char* msg, const char* delim, unsigned int fwid)
{
    LOG(info) << msg ; 
    typedef std::vector<std::string> VS ; 
    for(unsigned int i=0 ; i < getNumKeys() ; i++)
    {
        const char* key = getKey(i);
        VS elem ; 
        boost::split(elem, key, boost::is_any_of(delim));

        std::stringstream ss ; 
        for(VS::const_iterator it=elem.begin() ; it != elem.end() ; it++)
             ss << std::setw(fwid) << *it << " " ; 

        LOG(info) << ss.str() ; 
    }
}

void GItemList::replaceField(unsigned int field, const char* from, const char* to, const char* delim)
{
    typedef std::vector<std::string> VS ; 
    for(unsigned int i=0 ; i < getNumKeys() ; i++)
    {
        unsigned int changes = 0 ; 
        const char* key = getKey(i);
        VS elem ; 
        boost::split(elem, key, boost::is_any_of(delim));
        if(field >= elem.size()) continue ;  

        if(elem[field].compare(from) == 0)
        {
            elem[field] = to ; 
            changes += 1 ;  
        }


        if(changes > 0)
        {
            std::string edited = boost::algorithm::join(elem, delim);
            setKey(i, edited.c_str());
        }

    }
}

GItemList* GItemList::make_slice(const char* slice_)
{
    NSlice* slice = slice_ ? new NSlice(slice_) : NULL ;
    return make_slice(slice);
}

GItemList* GItemList::make_slice(NSlice* slice)
{
    GItemList* spawn = new GItemList( m_itemtype.c_str(), m_reldir.c_str() );
    spawn->setOrder(m_order);

    unsigned int ni = getNumKeys();
    if(!slice)
    {   
        slice = new NSlice(0, ni, 1); 
        LOG(warning) << "GItemList::make_slice NULL slice, defaulting to full copy " << slice->description() ;
    }   
    unsigned int count = slice->count();

    LOG(info) << "GItemList::make_slice from " 
              << ni << " -> " << count 
              << " slice " << slice->description() ;

    assert(count <= ni);

    for(unsigned int i=slice->low ; i < slice->high ; i+=slice->step)
    {   
         spawn->add(getKey(i));
    }   
    return spawn ; 
}


void GItemList::add(GItemList* other)
{
    unsigned int ok = other->getNumKeys();
    for(unsigned int i=0 ; i < ok ; i++) add(other->getKey(i));
}


