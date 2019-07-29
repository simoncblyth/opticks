
#include "BLog.hh"
#include "BTxt.hh"
#include "BStr.hh"
#include "SVec.hh"
#include "PLOG.hh"


const double BLog::TOLERANCE = 1e-6 ; 


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
    :
    m_sequence(NULL)
{
}
void BLog::setSequence(const std::vector<double>*  sequence)
{
    m_sequence = sequence ;
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

int BLog::getSequenceIndex(unsigned i) const 
{
    double s = getValue(i); 
    return m_sequence ? SVec<double>::FindIndexOfValue( *m_sequence, s, TOLERANCE) : -1 ;  
}


const std::vector<double>& BLog::getValues() const 
{
    return m_values ; 
}



void BLog::dump(const char* msg) const 
{
    LOG(info) << msg ; 
    assert( m_keys.size() == m_values.size() ) ; 
    for( unsigned i=0 ; i < m_keys.size() ; i++ ) 
    {
        int idx = getSequenceIndex(i) ;   
        std::cerr 
             << std::setw(30) << m_keys[i]
             << ":" 
             << std::setw(4) << idx
             << " "  
             << ":" 
             << std::setw(10) << std::fixed << std::setprecision(9) << m_values[i]
             << std::endl 
             ;
    }
}

std::string BLog::makeLine(unsigned i) const 
{
    std::stringstream ss ; 
    ss << "u_" 
       << m_keys[i]  
       << ":"
       << std::setprecision(9) << m_values[i]
       << " " 
       ; 
    return ss.str(); 
}


BTxt* BLog::makeTxt() const 
{
    BTxt* txt = new BTxt ; 
    for( unsigned i=0 ; i < m_keys.size() ; i++ ) 
    {
        std::string line = makeLine(i); 
        txt->addLine(line); 
    }
    return txt ; 
}

void BLog::write(const char* path) const 
{
    BTxt* t = makeTxt(); 
    t->write(path); 
}



int BLog::Compare( const BLog* a , const BLog* b )
{
    unsigned ai = a->getNumKeys() ; 
    unsigned bi = b->getNumKeys() ; 
    unsigned ni = std::max( ai, bi ); 

    int RC = 0 ; 

    for(unsigned i=0 ; i < ni ; i++)
    {
         int rc = 0 ;  

         const char* ak = i < ai ? a->getKey(i) : NULL ; 
         const char* bk = i < bi ? b->getKey(i) : NULL ; 
         bool mk = ak && bk && strcmp(ak, bk) == 0 ; 
         if( !mk ) rc |= 0x1 ;   


         double av      = i < ai ? a->getValue(i) : -1. ; 
         double bv      = i < bi ? b->getValue(i) : -1. ; 
         double dv      = av - bv ; 
         bool mv = std::abs(dv) < TOLERANCE ; 
         if( !mv ) rc |= 0x10 ;   

         int ax = a->getSequenceIndex(i) ;   
         int bx = b->getSequenceIndex(i) ;   

         const char* marker = rc == 0 ? " " : "*" ; 

         std::cerr
              << " i " << std::setw(4) << i 
              << " rc " << std::setw(4) << std::hex << rc << std::dec 
              << " ak/bk " 
              << std::setw(40) << std::right << ak 
              << "/"
              << std::setw(40) << std::left << bk << std::right 
              << "  "
              << std::setw(1) << marker 
              << "  "
              << " ax/bx " 
              << std::setw(2)  << ax
              << "/"
              << std::setw(2)  << bx
              << "   " 
              << " av/bv " 
              << std::setw(12) << std::setprecision(10) << av
              << "/"
              << std::setw(12) << std::setprecision(10) << bv
              << "   " 
              << " dv " << std::setw(13) << std::setprecision(10) << dv
              << std::endl 
              ; 

         RC |= rc ;  
    }

    LOG(info) 
        << " ai " << ai  
        << " bi " << bi 
        << " RC " << RC 
        << " tol " << std::setw(13) << std::setprecision(10) << TOLERANCE
        ;

    return RC ; 
}



