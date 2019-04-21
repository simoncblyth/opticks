#include <iostream>
#include <sstream>

#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>

#include "BTimesTable.hh"
#include "BTimes.hh"

#include "PLOG.hh"

BTimesTable::BTimesTable(const std::vector<std::string>& columns)
    :
    m_tx(NULL),
    m_ty(NULL),
    m_tz(NULL),
    m_tw(NULL),
    m_label(NULL)
{
    init(columns);
}

BTimesTable::BTimesTable(const char* cols, const char* delim)
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


void BTimesTable::init(const std::vector<std::string>& columns)
{
    unsigned numcol = columns.size() ;
    for(unsigned int j=0 ; j < numcol ; j++) m_table.push_back(new BTimes(columns[j].c_str()));

    m_tx = numcol > 0 ? getColumn(0) : NULL  ; 
    m_ty = numcol > 1 ? getColumn(1) : NULL  ; 
    m_tz = numcol > 2 ? getColumn(2) : NULL  ; 
    m_tw = numcol > 3 ? getColumn(3) : NULL  ; 
}


unsigned BTimesTable::getNumColumns()
{
    return m_table.size() ; 
}
BTimes* BTimesTable::getColumn(unsigned int j)
{
    unsigned numcol = getNumColumns();
    return j < numcol ? m_table[j] : NULL ; 
}
std::vector<std::string>& BTimesTable::getLines()
{
    makeLines(); 
    return m_lines ;  
}


template <typename T>
const char* BTimesTable::makeLabel( T row_, int count )
{
    std::stringstream ss ; 
    ss << boost::lexical_cast<std::string>(row_) ;
    if(count > -1) ss << "_" << count ; 

    std::string slabel = ss.str();
    return strdup(slabel.c_str());
}


template <typename T>
void BTimesTable::add( T row_, double x, double y, double z, double w, int count )
{
    const char* label = makeLabel( row_ , count );
    setLabel(label);

    if(m_tx) m_tx->add(m_label, x );
    if(m_ty) m_ty->add(m_label, y );
    if(m_tz) m_tz->add(m_label, z );
    if(m_tw) m_tw->add(m_label, w );
} 

void BTimesTable::setLabel(const char* label)
{
    free((void*)m_label);
    m_label = label ; 
}
const char* BTimesTable::getLabel()
{
    return m_label ;
}


void BTimesTable::dump(const char* msg, const char* startswith, const char* spacewith, double tcut )
{
    makeLines();
    LOG(info) << msg 
              << " filter: " << ( startswith ? startswith : "NONE" )
              ;

    assert(m_lines.size() == m_names.size());
    unsigned nline = m_lines.size();

    double prior_first(0);

    for( unsigned i=0 ; i < nline ; i++)
    {
        std::string line = m_lines[i];
        const char* name = m_names[i].c_str();
        double first = m_first[i] ;

        bool space = spacewith == NULL ? false : 
                        strlen(spacewith) <= strlen(name) && strncmp(name,spacewith,strlen(spacewith))==0 ;  

        bool include = startswith == NULL ? true  :
                        strlen(startswith) <= strlen(name) && strncmp(name,startswith,strlen(startswith))==0 ;  

        if(include) 
        {
            double delta = first - prior_first ; 
            if(space) std::cout << std::endl ; 

            bool cut = tcut == 0.0 ? false : delta < tcut ;   // suppress lines with delta less than tcut 

            if(!cut)
            std::cout << std::fixed << std::setw(15) << std::setprecision(3) << delta << " " << line << std::endl ;  

            prior_first = first ; 
        }
    }
}

void BTimesTable::save(const char* dir)
{
    for(unsigned int j=0 ; j < m_table.size() ; j++)
    {  
        BTimes* ts = m_table[j];
        ts->save(dir);
    }
}

void BTimesTable::load(const char* dir)
{
    for(unsigned int j=0 ; j < m_table.size() ; j++)
    {  
        BTimes* ts = m_table[j];
        ts->load(dir);
    }
}


void BTimesTable::makeLines()
{
    unsigned wid = 15 ; 

    m_lines.clear() ;  
    m_names.clear() ;  
    m_first.clear() ;  

    unsigned int nrow = 0 ;
    unsigned int numcol = m_table.size() ;

    std::stringstream ll ;  
    for(unsigned int j=0 ; j < numcol ; j++)
    {  
        BTimes* ts = m_table[j];
        if(nrow == 0)
            nrow = ts->getNumEntries();
        else
           assert(ts->getNumEntries() == nrow && "all times must have same number of entries" );

        ll << std::setw(wid) << ts->getLabel() ;
    }
    m_lines.push_back(ll.str());
    m_names.push_back("header");
    m_first.push_back(0);

    for(unsigned int i=0 ; i < nrow ; i++)
    {
        std::stringstream ss ;  

        std::string rowname ; 
        double first = 0 ; 
        for(unsigned int j=0 ; j < numcol ; j++)
        { 
            BTimes* ts = m_table[j];
            std::pair<std::string, double>& entry = ts->getEntry(i);

            if(rowname.empty()) 
                rowname = entry.first ; 
            else
                assert(entry.first.compare(rowname) == 0) ;

             ss << std::fixed << std::setw(wid) << std::setprecision(3) << entry.second ;

            if(j==0) first = entry.second ;
        }

        ss << " : " << rowname ;       

        m_lines.push_back(ss.str());
        m_names.push_back(rowname);
        m_first.push_back(first);
    }
}


template BRAP_API void BTimesTable::add(int           , double, double, double, double, int);
template BRAP_API void BTimesTable::add(unsigned      , double, double, double, double, int);
template BRAP_API void BTimesTable::add(char*         , double, double, double, double, int);
template BRAP_API void BTimesTable::add(const char*   , double, double, double, double, int);

template BRAP_API const char* BTimesTable::makeLabel( int          , int count );
template BRAP_API const char* BTimesTable::makeLabel( unsigned     , int count );
template BRAP_API const char* BTimesTable::makeLabel( char*        , int count );
template BRAP_API const char* BTimesTable::makeLabel( const char*  , int count );


