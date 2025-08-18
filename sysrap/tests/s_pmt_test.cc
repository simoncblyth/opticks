#include "s_pmt.h"
#include "NPFold.h"

struct s_pmt_test
{
     static int desc();
     static int lpmtidx();
     static NP* lpmtidx_array();

     static int oldcontiguousidx();
     static NP* oldcontiguousidx_array();

     static int contiguousidx();
     static NP* contiguousidx_array();

     static NPFold* makefold_();
     static int makefold();

     static int main();
};

int s_pmt_test::desc()
{
    std::cout << s_pmt::desc() ;
    return 0;
}

int s_pmt_test::lpmtidx()
{
    for(int i=0 ; i < s_pmt::NUM_CD_LPMT_AND_WP ; i++)
    {
        int lpmtidx = i ;
        int lpmtid = s_pmt::pmtid_from_lpmtidx(lpmtidx);
        int lpmtidx_1 = s_pmt::lpmtidx_from_pmtid(lpmtid);
        assert( lpmtidx_1 == lpmtidx );
    }
    return 0;
}

NP* s_pmt_test::lpmtidx_array()
{
    int ni = s_pmt::NUM_CD_LPMT_AND_WP ;
    int nj = 3 ;
    NP* a = NP::Make<int>( ni, nj );
    int* aa = a->values<int>();
    for(int i=0 ; i < ni ; i++)
    {
        int lpmtidx = i ;
        int lpmtid = s_pmt::pmtid_from_lpmtidx(lpmtidx);
        int contiguousidx = s_pmt::contiguousidx_from_pmtid(lpmtid);

        aa[i*nj+0] = lpmtidx ;
        aa[i*nj+1] = lpmtid ;
        aa[i*nj+2] = contiguousidx ;
    }
    a->labels = new std::vector<std::string> { "lpmtidx", "lpmtid", "contiguousidx" };
    return a ;
}


int s_pmt_test::oldcontiguousidx()
{
    for(int i=0 ; i < s_pmt::NUM_ALL ; i++)
    {
        int oldcontiguousidx = i ;
        int pmtid = s_pmt::pmtid_from_oldcontiguousidx(oldcontiguousidx);
        int oldcontiguousidx_1 = s_pmt::oldcontiguousidx_from_pmtid(pmtid);
        assert( oldcontiguousidx_1 == oldcontiguousidx );
    }
    return 0;
}

NP* s_pmt_test::oldcontiguousidx_array()
{
    int ni = s_pmt::NUM_ALL ;
    int nj = 3 ;
    NP* a = NP::Make<int>( ni, nj );
    int* aa = a->values<int>();
    for(int i=0 ; i < ni ; i++)
    {
        int oldcontiguousidx = i ;
        int pmtid = s_pmt::pmtid_from_oldcontiguousidx(oldcontiguousidx);
        int lpmtidx = s_pmt::lpmtidx_from_pmtid(pmtid);

        aa[i*nj+0] = oldcontiguousidx ;
        aa[i*nj+1] = pmtid ;
        aa[i*nj+2] = lpmtidx ;
    }

    a->labels = new std::vector<std::string> { "oldcontiguousidx", "pmtid", "lpmtidx" };

    return a ;
}







int s_pmt_test::contiguousidx()
{
    for(int i=0 ; i < s_pmt::NUM_ALL ; i++)
    {
        int contiguousidx = i ;
        int pmtid = s_pmt::pmtid_from_contiguousidx(contiguousidx);
        int contiguousidx_1 = s_pmt::contiguousidx_from_pmtid(pmtid);
        assert( contiguousidx_1 == contiguousidx );
    }
    return 0;
}

NP* s_pmt_test::contiguousidx_array()
{
    int ni = s_pmt::NUM_ALL ;
    int nj = 3 ;
    NP* a = NP::Make<int>( ni, nj );
    int* aa = a->values<int>();
    for(int i=0 ; i < ni ; i++)
    {
        int contiguousidx = i ;
        int pmtid = s_pmt::pmtid_from_contiguousidx(contiguousidx);
        int lpmtidx = s_pmt::lpmtidx_from_pmtid(pmtid);

        aa[i*nj+0] = contiguousidx ;
        aa[i*nj+1] = pmtid ;
        aa[i*nj+2] = lpmtidx ;
    }

    a->labels = new std::vector<std::string> { "contiguousidx", "pmtid", "lpmtidx" };

    return a ;
}






NPFold* s_pmt_test::makefold_()
{
    NPFold* f = new NPFold ;
    f->add( "lpmtidx" , lpmtidx_array() );
    f->add( "oldcontiguousidx" , oldcontiguousidx_array() );
    f->add( "contiguousidx" , contiguousidx_array() );
    return f ;
}

int s_pmt_test::makefold()
{
    NPFold* f = makefold_();
    f->save("$FOLD");
    return 0;
}


int s_pmt_test::main()
{
    int rc = 0 ;
    rc += desc();
    rc += lpmtidx();
    rc += oldcontiguousidx();
    rc += contiguousidx();
    rc += makefold();
    return rc ;
}


int main()
{
   return s_pmt_test::main() ;
}
