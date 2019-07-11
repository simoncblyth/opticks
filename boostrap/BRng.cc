#include "PLOG.hh"
#include "BRng.hh"


const plog::Severity BRng::LEVEL = PLOG::EnvLevel("BRng", "DEBUG") ; 


float BRng::getLo() const 
{
    return m_lo ;
}
float BRng::getHi() const 
{
    return m_hi ;
}


BRng::BRng(float lo, float hi, unsigned _seed, const char* label) 
   :   
   m_lo(lo),
   m_hi(hi),
   m_rng(NULL),   //  cannot give a ctor arg to m_rng here ?
   m_dst(NULL),
   m_gen(NULL),
   m_label( label ? strdup(label) : "?" ),
   m_count(0)
{
    setSeed(_seed);
}


float BRng::one()
{   
    m_count++ ; 
    return (*m_gen)() ;
}


void BRng::two(float& a, float& b)
{   
    a = one(); 
    b = one();
}


void BRng::setSeed(unsigned _seed)
{
    LOG(LEVEL) << m_label << " setSeed(" << _seed << ")" ; 

    m_seed = _seed ; 

    // forced to recreate as trying to seed/reset 
    // existing ones failed to giving a fresh sequence

    delete m_gen ; 
    delete m_dst ; 
    delete m_rng ; 

    m_rng = new RNG_t(m_seed) ; 
    m_dst = new DST_t(m_lo, m_hi);
    m_gen = new GEN_t(*m_rng, *m_dst) ;

}

std::string BRng::desc() const 
{
    std::stringstream ss ; 

    ss << m_label 
       << " "
       << " seed " << m_seed
       << " lo " << m_lo
       << " hi " << m_hi
       << " count " << m_count
       ;

    return ss.str();
}

void BRng::dump()
{
    LOG(info) << desc() ; 
    for(unsigned i=0 ; i < 10 ; i++ ) 
       std::cout << one() << std::endl ; 
}






