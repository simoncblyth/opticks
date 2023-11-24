// name=std__unique_copy ; gcc $name.cc -std=c++11 -lstdc++ -o /tmp/$name && /tmp/$name

/**
https://stackoverflow.com/questions/28100712/better-way-of-counting-unique-item

https://cplusplus.com/reference/algorithm/unique_copy/
**/


// unique_copy example
#include <iostream>     // std::cout
#include <algorithm>    // std::unique_copy, std::sort, std::distance
#include <vector>       // std::vector

bool myfunction (int i, int j) {
  return (i==j);
}

int main () {
  int myints[] = {10,20,20,20,30,30,20,20,10};
  std::vector<int> myvector (9);                            // 0  0  0  0  0  0  0  0  0

  // using default comparison:
  std::vector<int>::iterator it;
  it=std::unique_copy (myints,myints+9,myvector.begin());   // 10 20 30 20 10 0  0  0  0
                                                            //                ^

  std::sort (myvector.begin(),it);                          // 10 10 20 20 30 0  0  0  0
                                                            //                ^

  // using predicate comparison:
  it=std::unique_copy (myvector.begin(), it, myvector.begin(), myfunction);
                                                            // 10 20 30 20 30 0  0  0  0
                                                            //          ^

  myvector.resize( std::distance(myvector.begin(),it) );    // 10 20 30

  // print out content:
  std::cout << "myvector contains:";
  for (it=myvector.begin(); it!=myvector.end(); ++it)
    std::cout << ' ' << *it;
  std::cout << '\n';

  return 0;
}



