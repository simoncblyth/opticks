#include "TimesTable.hpp"
#include "Times.hpp"
#include "BLog.hh"
#include <iostream>
#include <sstream>

#include <boost/algorithm/string.hpp>


TimesTable::TimesTable(const char* cols, const char* delim)
{
    std::vector<std::string> columns ; 
    boost::split(columns, cols, boost::is_any_of(delim));
    init(columns);
}

void TimesTable::init(const std::vector<std::string>& columns)
{
    for(unsigned int j=0 ; j < columns.size() ; j++) m_table.push_back(new Times(columns[j].c_str()));
}

void TimesTable::dump(const char* msg)
{
    makeLines();
    LOG(info) << msg ; 
    typedef std::vector<std::string>::const_iterator VSI ;  
    for(VSI it=m_lines.begin() ; it != m_lines.end() ; it++) std::cout << *it << std::endl ;  
}

void TimesTable::save(const char* dir)
{
    for(unsigned int j=0 ; j < m_table.size() ; j++)
    {  
        Times* ts = m_table[j];
        ts->save(dir);
    }
}

void TimesTable::load(const char* dir)
{
    for(unsigned int j=0 ; j < m_table.size() ; j++)
    {  
        Times* ts = m_table[j];
        ts->load(dir);
    }
}


void TimesTable::makeLines()
{
    m_lines.clear() ;  

    unsigned int nrow = 0 ;
    unsigned int ncol = m_table.size() ;

    for(unsigned int j=0 ; j < ncol ; j++)
    {  
        Times* ts = m_table[j];
        if(nrow == 0)
            nrow = ts->getNumEntries();
        else
           assert(ts->getNumEntries() == nrow && "all times must have same number of entries" );
    }

    for(unsigned int i=0 ; i < nrow ; i++)
    {
        std::stringstream ss ;  

        std::string rowname ; 
        for(unsigned int j=0 ; j < ncol ; j++)
        { 
            Times* ts = m_table[j];
            std::pair<std::string, double>& entry = ts->getEntry(i);

            if(rowname.empty()) 
                rowname = entry.first ; 
            else
                assert(entry.first.compare(rowname) == 0) ;

             ss << std::fixed << std::setw(15) << std::setprecision(3) << entry.second ;
        }

        ss << " : " << rowname ;       

        m_lines.push_back(ss.str());
    }
}


