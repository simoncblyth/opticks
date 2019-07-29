
#include "BLog.hh"
#include "BTxt.hh"
#include "BStr.hh"
#include "PLOG.hh"

BLog* BLog::Load(const char* path)
{
    BTxt* txt = BTxt::Load(path); 
    BLog* log = new BLog ; 
    LOG(info) << txt->desc();  
    unsigned ni = txt->getNumLines(); 
    for(unsigned i=0 ; i < ni ; i++)
    {
        const std::string& line = txt->getString(i); 
        std::string k, v ;  
        if(0==ParseLine(line, k, v ))
        { 
            double vf = BStr::atod(v.c_str(), 0. ); 
            log->add( k.c_str(),  vf ); 
        }
    }
    return log ; 
}


int BLog::ParseLine( const std::string& line,  std::string& k, std::string& v )
{
    std::size_t pu = line.find("u_") ; 
    if( pu == std::string::npos ) return 1  ;
    pu += 2 ;     

    std::size_t pc = line.find(":", pu) ; 
    if( pc == std::string::npos ) return 2 ;   
    pc += 1 ;  

    std::size_t ps = line.find(" ", pc) ; 
    if( ps == std::string::npos ) return 3 ;   

    k = line.substr(pu,pc-pu-1); 
    v = line.substr(pc,ps-pc); 
    return 0 ;  
}



BLog::BLog()
{
}

void BLog::add(const char* key, double value )
{
    m_keys.push_back(key); 
    m_values.push_back(value); 
}

unsigned BLog::getNumKeys() const 
{
    return m_keys.size();  
}
const char* BLog::getKey(unsigned i) const 
{
    return m_keys[i].c_str(); 
}
double BLog::getValue(unsigned i) const 
{
    return m_values[i] ; 
}

void BLog::dump(const char* msg) const 
{
    LOG(info) << msg ; 
    assert( m_keys.size() == m_values.size() ) ; 
    for( unsigned i=0 ; i < m_keys.size() ; i++ ) 
    {
        std::cerr 
             << std::setw(30) << m_keys[i]
             << ":" 
             << std::setw(10) << std::fixed << std::setprecision(9) << m_values[i]
             << std::endl 
             ;
    }
}




