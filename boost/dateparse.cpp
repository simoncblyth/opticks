/* 
*/

#include <iostream>
#include <boost/date_time/local_time/local_time.hpp>

int main() 
{
    using namespace boost::gregorian;
    using namespace boost::posix_time;
    using namespace boost::local_time;
    using namespace std;

    try {

      stringstream ss;

      //date d(2004, Feb, 29);
      //time_duration td(12,34,56,789);
      //ss << d << ' ' << td;

      local_time_facet* output_facet = new local_time_facet();
      output_facet->format("%a %b %d, %H:%M %z");
      local_time_input_facet* input_facet = new local_time_input_facet();
      input_facet->format("%b %d, %Y %H:%M:%S");

      ss.imbue(locale(locale::classic(), output_facet));
      ss.imbue(locale(ss.getloc(), input_facet));

      ptime pt(not_a_date_time);
      cout << pt << endl;
      ss >> pt ;
      cout << pt << endl ; 

      local_date_time ldt(not_a_date_time);

      ss.str("Mar 09, 2010 12:55:10");

      ss << ldt;

      cout << ldt << endl; // cout is using default format "2005-May-08 12:15:00 UTC"

      
   }
    catch(exception& e) {
      cout << "  Exception: " <<  e.what() << endl;
    }


    return 0;
}
