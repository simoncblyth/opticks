#include "SPMT.h"
#include "SPMTAccessor.h"
#include <map>
#include <unordered_map>

struct SPMTAccessor_test
{
    SPMTAccessor_test( SPMTAccessor* accessor_ ) : accessor(accessor_) {} ;

    SPMTAccessor* accessor ;

    NP*  get_pmtcat_qescale() const ;
    NP*  get_pmtid_qe() const ;
    NP*  get_stackspec() const ;

    void get_stackspec_one() const ;


    static void CategoriesPopulate( std::unordered_map<int, int>& pmt_categories );
    static void CategoriesCount( std::map<int, int>& cat_count, std::unordered_map<int, int>& pmt_categories );
    static std::string CategoriesDesc( std::map<int, int>& cat_count );
    static int Categories();

};


NP* SPMTAccessor_test::get_pmtcat_qescale() const
{
    int num_lpmt = accessor->get_num_lpmt();
    std::map<int, int> cats ;

    int ni = num_lpmt ;
    int nj = 2 ;

    NP* a = NP::Make<double>(ni, nj );
    double* aa = a->values<double>();

    for(int i=0 ; i < ni ; i++)
    {
        int pmtid = i ;
        int pmtcat = accessor->get_pmtcat(pmtid);
        double qs = accessor->get_qescale(pmtid);

        aa[i*nj+0] = pmtcat ;
        aa[i*nj+1] = qs ;

        if(i % 1000 == 0) std::cout
            << std::setw(6) << pmtid
            << " : "
            << std::setw(2) << pmtcat
            << " : "
            << qs
            << std::endl
            ;

        cats[pmtcat] += 1 ;
    }

    typedef std::map<int,int>  MII ;

    int total = 0 ;
    for(MII::const_iterator it=cats.begin() ; it != cats.end() ; it++)
    {
        std::cout << it->first << " : " << it->second << std::endl ;
        total += it->second ;
    }
    std::cout << " total " << total << std::endl ;
    return a ;
}

NP* SPMTAccessor_test::get_pmtid_qe() const
{
    int num_lpmt = accessor->get_num_lpmt();

    int ni = num_lpmt ;
    int nj = 100 ;
    int nk = 2 ;

    NP* a = NP::Make<double>(ni, nj, nk );
    double* aa = a->values<double>();

    for(int i=0 ; i < ni ; i++)
    {
        int pmtid = i ;
        for(int j=0 ; j < nj ; j++)
        {
            double energy = SPMT::GetValueInRange(j, nj, 1.55, 15.5 );
            double qe = accessor->get_pmtid_qe(pmtid, energy );

            aa[i*nj*nk+j*nk+0] = energy ;
            aa[i*nj*nk+j*nk+1] = qe ;

            if(i % 1000 == 0 && j == nj/2 )  std::cout
                << std::setw(6) << pmtid
                << " : "
                << qe
                << std::endl
                ;
        }
    }
    return a ;
}


NP* SPMTAccessor_test::get_stackspec() const
{
    int num_lpmt = accessor->get_num_lpmt();

    int ni = num_lpmt ;
    int nj = 100 ;
    int nk = 16 ;

    NP* a = NP::Make<double>(ni, nj, nk );
    double* aa = a->values<double>();

    for(int i=0 ; i < ni ; i++)
    {
        int pmtid = i ;
        int pmtcat = accessor->get_pmtcat(pmtid);
        for(int j=0 ; j < nj ; j++)
        {
            double energy = SPMT::GetValueInRange(j, nj, 1.55, 15.5 );
            std::array<double, 16> ss ;
            accessor->get_stackspec( ss, pmtcat, energy );
            for(int k=0 ; k < nk ; k++) aa[i*nj*nk+j*nk+k] = ss[k] ;
        }
    }
    return a ;
}



void SPMTAccessor_test::get_stackspec_one() const
{
    int pmtcat = 0 ;
    double energy = 4. ;
    std::array<double, 16> ss ;
    accessor->get_stackspec( ss, pmtcat, energy );

    std::cout << SPMTAccessor::Desc(ss) ;
}



void SPMTAccessor_test::CategoriesPopulate( std::unordered_map<int, int>& pmt_categories )
{
    pmt_categories[0] = 0 ;
    pmt_categories[1] = 0 ;
    pmt_categories[2] = 0 ;
    pmt_categories[3] = 0 ;

    pmt_categories[10] = 1 ;
    pmt_categories[11] = 1 ;
    pmt_categories[12] = 1 ;
    pmt_categories[13] = 1 ;
    pmt_categories[14] = 1 ;

    pmt_categories[100] = 2 ;
    pmt_categories[110] = 2 ;
    pmt_categories[120] = 2 ;
    pmt_categories[130] = 2 ;
    pmt_categories[140] = 2 ;
    pmt_categories[150] = 2 ;
}


void SPMTAccessor_test::CategoriesCount( std::map<int, int>& cat_count, std::unordered_map<int, int>& pmt_categories )
{
    for( auto it : pmt_categories ) cat_count[it.second]++ ;
}


std::string SPMTAccessor_test::CategoriesDesc( std::map<int, int>& cat_count )
{
    std::stringstream ss ;
    for( auto it : cat_count ) ss
          << std::setw(3) << it.first
          << " : "
          << std::setw(6) << it.second
          << std::endl
          ;

    std::string str = ss.str() ;
    return str ;
}

int SPMTAccessor_test::Categories()
{
    std::unordered_map<int,int> pmt_categories ;
    CategoriesPopulate( pmt_categories );

    std::map<int,int> catmap ;
    CategoriesCount( catmap, pmt_categories );

    std::cout << CategoriesDesc( catmap );

    return 0;
}


int main(int argc, char** argv)
{
    SPMTAccessor_test::Categories();

    SPMT* pmt = SPMT::CreateFromJPMT();
    if(pmt == nullptr) return 1 ;

    SPMTAccessor* accessor = new SPMTAccessor(pmt);
    assert( accessor );

    SPMTAccessor_test test(accessor);
    test.get_stackspec_one() ;

    NPFold* f = new NPFold ;
    f->add( "get_pmtcat_qescale", test.get_pmtcat_qescale() );
    f->add( "get_pmtid_qe",       test.get_pmtid_qe() );
    f->add( "get_stackspec",      test.get_stackspec() );
    f->save("$FOLD");

    return 0 ;
}
