#include "Sparse.hh"

#include "NPY.hpp"
#include "NLog.hpp"

#include <numeric>

template <typename T>
void Sparse<T>::count_unique()
{
    typedef std::vector<T> V ;  
    V& data = m_npy->data();

    std::sort(data.begin(), data.end());

    m_num_unique = std::inner_product(
                                  data.begin(),data.end() - 1,   // first1, last1  
                                             data.begin() + 1,   // first2
                                                       int(1),   // output type init 
                                             std::plus<int>(),   // reduction operator
                                       std::not_equal_to<T>()    // pair-by-pair operator, returning 1 at edges 
                                      );  

    LOG(info) << "Sparse<T>::count_unique"
              << " label " << m_label
              << " num_unique " << m_num_unique 
              ;

    reduce_by_key();
    sort_by_key();

}

template <typename T>
void Sparse<T>::reduce_by_key()
{
    m_valuecount.resize(m_num_unique);

    typedef std::vector<T> V ;  
    V& data = m_npy->data();


    typedef std::pair<T, int> P ; 
 
    unsigned int n = data.size();
    T* vals = data.data() ; 
    //T prev = std::numeric_limits<T>::max();
    T prev = *(vals + 0);
    int count(1) ; 
    unsigned int unique(0) ;

    T value ; 

    // from the second
    for(unsigned int i=1 ; i < n ; i++)
    {
        value = *(vals+i) ;
        if(value == prev)
        {
            count += 1 ; 
        }
        else 
        {
            m_valuecount[unique++] = P(value, count) ;
            prev = value ; 
            count = 1 ; 
        } 
    }
 
    // no transition possible, so must special case the last
    T last = *(vals+n-1) ;
    if(last == prev) 
    {
        m_valuecount[unique++] = P(value, count) ;
    }


    LOG(info) << "Sparse<T>::reduce_by_key"
              << " unique " << unique 
              << " num_unique " << m_num_unique 
              ;

    assert(unique == m_num_unique);
}


/*

Getting expected counts but not keys...


[2016-May-31 16:13:12.050778]:info: indexSequence seqhis : num_unique 48
[ 0]             4bcd    340270
[ 1]              86d    107598
[ 2]            46ccd     23218
[ 3]              46d     18866
[ 4]              8bd      3179
[ 5]              7cd      2204
[ 6]             76cd      1696
[ 7]              8cd      1446
[ 8]             8ccd       382
[ 9]            8c66d       260
[10]              4bd       197
[11]           8cbc6d       190
[12]              4cd       132
[13]             866d       111
[14]             86bd        35
[15]           4cbbcd        31
[16]        8cbc66ccd        31
[17]             8c6d        26
[18]           866ccd        24
[19]             8b6d        19
[20]            4bbcd        17
[21]       cbccbc6ccd         7
[22]       8cccccc6bd         7
[23]       cccccc6ccd         6
[24]          8cbc6bd         5
[25]           8b6ccd         4
[26]             4ccd         3
[27]           8cbbcd         3
[28]            8c6cd         3
[29]         8cbc6ccd         3
[30]            8cc6d         3
[31]          86cbbcd         2
[32]            4cc6d         2
[33]             7c6d         2
[34]       cbccbbbbcd         2
[35]           8cc66d         2
[36]             4c6d         2
[37]            86ccd         2
[38]       4bbbcc6ccd         1
[39]       8ccccc6ccd         1
[40]          4cc6ccd         1
[41]       bbbbcc6ccd         1
[42]         4cbc6ccd         1
[43]          8cc6ccd         1
[44]       cbcccc6ccd         1
[45]       ccbccc6ccd         1
[46]       cccccc6ccd         1
[47]        8cccc6ccd         1

In [4]: evt.history.table
Out[4]: 
                      4:PmtInBox 
                 8cd        0.681         340270       [3 ] TO BT SA
                 7cd        0.215         107598       [3 ] TO BT SD
                8ccd        0.046          23218       [4 ] TO BT BT SA
                  4d        0.038          18866       [2 ] TO AB
                 86d        0.006           3179       [3 ] TO SC SA
                 4cd        0.004           2204       [3 ] TO BT AB
                4ccd        0.003           1696       [4 ] TO BT BT AB
                 8bd        0.003           1446       [3 ] TO BR SA
                8c6d        0.001            382       [4 ] TO SC BT SA
               86ccd        0.001            260       [5 ] TO BT BT SC SA
                 46d        0.000            197       [3 ] TO SC AB
              8cbbcd        0.000            190       [6 ] TO BT BR BR BT SA
                 4bd        0.000            132       [3 ] TO BR AB
                7c6d        0.000            111       [4 ] TO SC BT SD
                866d        0.000             35       [4 ] TO SC SC SA
            8cbc6ccd        0.000             31       [8 ] TO BT BT SC BT BR BT SA
               8cc6d        0.000             31       [5 ] TO SC BT BT SA
                8b6d        0.000             26       [4 ] TO SC BR SA
              4cbbcd        0.000             24       [6 ] TO BT BR BR BT AB
                86bd        0.000             19       [4 ] TO BR SC SA



*/




template <class T>
struct second_descending : public std::binary_function<T,T,bool> {
    bool operator()(const T& a, const T& b) const {
        return a.second > b.second ;
    }
};

template <typename T>
void Sparse<T>::sort_by_key()
{
    typedef std::pair<T, int> P ; 
    std::sort( m_valuecount.begin(), m_valuecount.end(), second_descending<P>());
}



template <typename T>
std::string Sparse<T>::dump_(const char* msg) const
{
    std::stringstream ss ;
    ss << msg << " : num_unique " << m_num_unique << std::endl ;

    typedef std::pair<T, int> P ;
    for(unsigned int  i=0 ; i < m_valuecount.size() ; i++)
    {
        P valuecount = m_valuecount[i] ;

        ss << "[" << std::setw(2) << i << "] "  ;
        if(m_hexkey) ss << std::hex ;
        ss << std::setw(16) << valuecount.first ;
        if(m_hexkey) ss << std::dec ;
        ss << std::setw(10) << valuecount.second ;
        ss << std::endl ;
    }

    return ss.str();
}


template <typename T> 
void Sparse<T>::dump(const char* msg) const 
{
    LOG(info) << dump_(msg) ; 
}



template class Sparse<unsigned long long> ;

