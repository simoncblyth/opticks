#include <iostream>
#include <sstream>

#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>

#include "TimesTable.hpp"
#include "Times.hpp"

#include "PLOG.hh"

TimesTable::TimesTable(const std::vector<std::string>& columns)
    :
    m_tx(NULL),
    m_ty(NULL),
    m_tz(NULL),
    m_tw(NULL)
{
    init(columns);
}

TimesTable::TimesTable(const char* cols, const char* delim)
    :
    m_tx(NULL),
    m_ty(NULL),
    m_tz(NULL),
    m_tw(NULL)
{
    std::vector<std::string> columns ; 
    boost::split(columns, cols, boost::is_any_of(delim));
    init(columns);
}


void TimesTable::init(const std::vector<std::string>& columns)
{
    unsigned numcol = columns.size() ;
    for(unsigned int j=0 ; j < numcol ; j++) m_table.push_back(new Times(columns[j].c_str()));

    m_tx = numcol > 0 ? getColumn(0) : NULL  ; 
    m_ty = numcol > 1 ? getColumn(1) : NULL  ; 
    m_tz = numcol > 2 ? getColumn(2) : NULL  ; 
    m_tw = numcol > 3 ? getColumn(3) : NULL  ; 
}


unsigned TimesTable::getNumColumns()
{
    return m_table.size() ; 
}
Times* TimesTable::getColumn(unsigned int j)
{
    unsigned numcol = getNumColumns();
    return j < numcol ? m_table[j] : NULL ; 
}
std::vector<std::string>& TimesTable::getLines()
{
    makeLines(); 
    return m_lines ;  
}



template <typename T>
void TimesTable::add( T row_, double x, double y, double z, double w )
{
    std::string srow = boost::lexical_cast<std::string>(row_) ; 
    const char* row = srow.c_str() ; 

    if(m_tx) m_tx->add(row, x );
    if(m_ty) m_ty->add(row, y );
    if(m_tz) m_tz->add(row, z );
    if(m_tw) m_tw->add(row, w );
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

    unsigned wid = 15 ; 

    m_lines.clear() ;  

    unsigned int nrow = 0 ;
    unsigned int numcol = m_table.size() ;

    std::stringstream ll ;  
    for(unsigned int j=0 ; j < numcol ; j++)
    {  
        Times* ts = m_table[j];
        if(nrow == 0)
            nrow = ts->getNumEntries();
        else
           assert(ts->getNumEntries() == nrow && "all times must have same number of entries" );

        ll << std::setw(wid) << ts->getLabel() ;
    }
    m_lines.push_back(ll.str());

    for(unsigned int i=0 ; i < nrow ; i++)
    {
        std::stringstream ss ;  

        std::string rowname ; 
        for(unsigned int j=0 ; j < numcol ; j++)
        { 
            Times* ts = m_table[j];
            std::pair<std::string, double>& entry = ts->getEntry(i);

            if(rowname.empty()) 
                rowname = entry.first ; 
            else
                assert(entry.first.compare(rowname) == 0) ;

             ss << std::fixed << std::setw(wid) << std::setprecision(3) << entry.second ;
        }

        ss << " : " << rowname ;       

        m_lines.push_back(ss.str());
    }
}


template NPY_API void TimesTable::add(int           , double, double, double, double );
template NPY_API void TimesTable::add(unsigned      , double, double, double, double );
template NPY_API void TimesTable::add(char*         , double, double, double, double );
template NPY_API void TimesTable::add(const char*   , double, double, double, double );

