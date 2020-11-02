/*
 * Copyright (c) 2019 Opticks Team. All Rights Reserved.
 *
 * This file is part of Opticks
 * (see https://bitbucket.org/simoncblyth/opticks).
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software 
 * distributed under the License is distributed on an "AS IS" BASIS, 
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
 * See the License for the specific language governing permissions and 
 * limitations under the License.
 */

#include "GItemList.hh"

#include <climits>
#include <iostream>
#include <fstream>
#include <ostream>   
#include <algorithm>
#include <iterator>
#include <iomanip>


#include "SStr.hh"
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/join.hpp>


#include "BFile.hh"
#include "NSlice.hpp"

#include "PLOG.hh"


const char* GItemList::GITEMLIST = "GItemList" ; 

unsigned int GItemList::UNSET = UINT_MAX ; 


bool GItemList::isUnset(unsigned int index)
{
    return index == UNSET ; 
}


GItemList* GItemList::Load(const char* idpath, const char* itemtype, const char* reldir)
{
    GItemList* gil = new GItemList(itemtype, reldir);
    gil->load_(idpath); 
    return gil ;
}

void GItemList::load_(const char* idpath)
{
   std::string txtname = m_itemtype + ".txt" ;  

   const char* reldir = m_reldir.empty() ? NULL : m_reldir.c_str() ; 

   std::string txtpath = BFile::FormPath(idpath, reldir, txtname.c_str()) ;

   if(BFile::ExistsFile(txtpath.c_str()))
   {
       read_(txtpath.c_str());  
   } 
   else
   {
       LOG(error)
           << " MISSING ITEMLIST TXT " 
           << " txtpath " << txtpath 
           << " txtname " << txtname 
           << " reldir " << reldir
           << " m_itemtype " << m_itemtype
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
    const char* reldir = m_reldir.empty() ? NULL : m_reldir.c_str() ; 
    std::string txtpath = BFile::preparePath(idpath, reldir, txtname.c_str() ); 
    LOG(debug) << "GItemList::save writing to " << txtpath ;       
    save_(txtpath.c_str());
}

void GItemList::save(const char* idpath, const char* reldir, const char* txtname)
{
    std::string txtpath = BFile::preparePath(idpath, reldir, txtname ); 
    LOG(debug) << "GItemList::save writing to " << txtpath ;       
    save_(txtpath.c_str());
}


void GItemList::save_(const char* txtpath)
{
    std::ofstream ofs(txtpath);
    std::copy(m_list.begin(),m_list.end(),std::ostream_iterator<std::string>( ofs,"\n"));
    ofs.close();
}


GItemList* GItemList::Repeat( const char* itemtype, const char* name, unsigned numRepeats, const char* reldir )
{
   // HMM this is fine for small numbers of repeats, will need to rethink for large scale...
    GItemList* ls = new GItemList(itemtype, reldir );
    for(unsigned i=0 ; i < numRepeats ; i++ ) ls->add(name) ;
    return ls ; 
}


const std::string& GItemList::getRelDir() const 
{
    return m_reldir ; 
}


GItemList::GItemList(const char* itemtype, const char* reldir) : NSequence()
{
    // NULL reldir defaults to GITEMLIST, empty reldir "" results in no reldir

    m_itemtype = itemtype ? strdup(itemtype) : NULL ; 
    m_reldir   = reldir ? reldir : GITEMLIST ; 
}

void GItemList::add(const char* name)
{
    assert(name) ; 
    m_list.push_back(name);
}

unsigned int GItemList::getNumItems()
{
    return m_list.size();
}
unsigned int GItemList::getNumKeys() const 
{
    return m_list.size();
}

const char* GItemList::getKey(unsigned index) const 
{
    return index < m_list.size() ? m_list[index].c_str() : NULL  ;
}

unsigned GItemList::getNumUniqueKeys() const 
{
    std::vector<std::string> unames ; 
    for( unsigned i=0 ; i < m_list.size() ; i++)
    {
        std::string key = m_list[i] ; 
        if(std::find( unames.begin(), unames.end() , key) == unames.end() ) unames.push_back(key) ; 
    }
    return unames.size() ; 
}



void GItemList::getIndicesWithKeyEnding( std::vector<unsigned>& indices, const char* ending ) const 
{  
    for(unsigned i=0 ; i < m_list.size() ; i++)
    {
        const std::string& k = m_list[i] ; 
        if( SStr::EndsWith( k.c_str(), ending )) indices.push_back(i) ; 
    }
}

void GItemList::getIndicesWithKey( std::vector<unsigned>& indices, const char* key ) const 
{  
    for(unsigned i=0 ; i < m_list.size() ; i++)
    {
        const std::string& k = m_list[i] ; 
        if( strcmp( k.c_str(), key ) == 0) indices.push_back(i) ; 
    }
}


int GItemList::findIndexWithKeyStarting( const char* starting ) const 
{  
    int result(-1); 
    for(unsigned i=0 ; i < m_list.size() ; i++)
    {
        const std::string& k = m_list[i] ; 
        if( SStr::StartsWith( k.c_str(), starting )) 
        {
            result = i ; 
            break ; 
        }   
    }
    return result ; 
}


unsigned int GItemList::getIndex(const char* key) const 
{
    if(key)
    {
        for(unsigned int i=0 ; i < m_list.size() ; i++) if(m_list[i].compare(key) == 0) return i ;  
    } 
    return UNSET  ; 
}

int GItemList::findIndex(const char* key) const 
{
    int index(-1) ;  
    for(unsigned i=0 ; i < m_list.size() ; i++) 
    {
        if(m_list[i].compare(key) == 0) 
        {
           index = int(i); 
           break ; 
        }
    } 
    return index  ; 
}


void GItemList::setKey(unsigned int index, const char* newkey)
{
    if(index < m_list.size()) m_list[index] = newkey  ; 
}


void GItemList::setOrder(std::map<std::string, unsigned int>& order)
{
    m_order = order ; 
}


void GItemList::getCurrentOrder( std::map<std::string, unsigned int>& order )
{
    for(unsigned i=0 ; i < m_list.size() ; i++)
    {
        const std::string& k = m_list[i] ; 
        order[k] = i ; 
    }
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


