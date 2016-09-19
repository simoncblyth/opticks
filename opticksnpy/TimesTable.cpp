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
    m_tw(NULL),
    m_label(NULL)
{
    init(columns);
}

TimesTable::TimesTable(const char* cols, const char* delim)
    :
    m_tx(NULL),
    m_ty(NULL),
    m_tz(NULL),
    m_tw(NULL),
    m_label(NULL)
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
const char* TimesTable::makeLabel( T row_, int count )
{
    std::stringstream ss ; 
    ss << boost::lexical_cast<std::string>(row_) ;
    if(count > -1) ss << "_" << count ; 

    std::string slabel = ss.str();
    return strdup(slabel.c_str());
}


template <typename T>
void TimesTable::add( T row_, double x, double y, double z, double w, int count )
{
    const char* label = makeLabel( row_ , count );
    setLabel(label);

    if(m_tx) m_tx->add(m_label, x );
    if(m_ty) m_ty->add(m_label, y );
    if(m_tz) m_tz->add(m_label, z );
    if(m_tw) m_tw->add(m_label, w );
} 

void TimesTable::setLabel(const char* label)
{
    free((void*)m_label);
    m_label = label ; 
}
const char* TimesTable::getLabel()
{
    return m_label ;
}




void TimesTable::dump(const char* msg, const char* startswith)
{
    makeLines();
    LOG(info) << msg 
              << " filter: " << ( startswith ? startswith : "NONE" )
              ;

    assert(m_lines.size() == m_names.size());
    unsigned nline = m_lines.size();

    for( unsigned i=0 ; i < nline ; i++)
    {
        std::string line = m_lines[i];
        const char* name = m_names[i].c_str();

        bool include = startswith == NULL ? true  :
                        strlen(startswith) <= strlen(name) && strncmp(name,startswith,strlen(startswith))==0 ;  

        if(include) std::cout << line << std::endl ;  
    }
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
    m_names.clear() ;  

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
    m_names.push_back("header");

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
        m_names.push_back(rowname);
    }
}


template NPY_API void TimesTable::add(int           , double, double, double, double, int);
template NPY_API void TimesTable::add(unsigned      , double, double, double, double, int);
template NPY_API void TimesTable::add(char*         , double, double, double, double, int);
template NPY_API void TimesTable::add(const char*   , double, double, double, double, int);

template NPY_API const char* TimesTable::makeLabel( int          , int count );
template NPY_API const char* TimesTable::makeLabel( unsigned     , int count );
template NPY_API const char* TimesTable::makeLabel( char*        , int count );
template NPY_API const char* TimesTable::makeLabel( const char*  , int count );


