/**
schi2_test.cc
=============

Compare boost chi2 p_value with an expansion from Gemini

**/


#include <cmath>
#include <boost/math/distributions/chi_squared.hpp>


struct schi2_test
{
    static double PValue_0(double c2obs, double ndof);
    static double PValue_1(double c2obs, double ndof);
};


double schi2_test::PValue_0(double c2obs, double ndof) {
    if (c2obs < 0.0 || ndof <= 0.0) return 1.0;

    // Special case for ndof == 2 which collapses beautifully
    if (std::abs(ndof - 2.0) < 1e-9) {
        return std::exp(-c2obs / 2.0);
    }

    double a = ndof / 2.0;
    double x = c2obs / 2.0;

    // Use a standard series expansion for the regularized incomplete gamma function
    if (x < a + 1.0) {
        // Series expansion for lower incomplete gamma
        double sum = 1.0 / a;
        double term = sum;
        for (int i = 1; i < 100; ++i) {
            term *= x / (a + i);
            sum += term;
            if (term < sum * 1e-15) break; // Convergence threshold
        }
        // P-value = 1 - CDF = 1 - (series_result)
        return 1.0 - (std::exp(-x + a * std::log(x) - std::lgamma(a)) * sum);
    } else {
        // Continued fraction for upper incomplete gamma (direct survival probability)
        double a0 = 1.0, a1 = x;
        double b0 = 0.0, b1 = 1.0;
        double fac = 1.0;
        double n = 1.0;
        double g = 0.0, gold = 0.0;

        for (int i = 1; i < 100; ++i) {
            double ana = n - a;
            a0 = (a1 + a0 * ana) * fac;
            b0 = (b1 + b0 * ana) * fac;
            double anf = n * fac;
            a1 = x * a0 + anf * a1;
            b1 = x * b0 + anf * b1;

            if (a1 != 0.0) {
                fac = 1.0 / a1;
                g = b1 * fac;
                if (std::abs(g - gold) < 1e-15) {
                    return std::exp(-x + a * std::log(x) - std::lgamma(a)) * g;
                }
                gold = g;
            }
            n += 1.0;
        }
        return 0.0; // Fallback for no convergence / extreme values
    }
}



double schi2_test::PValue_1(double c2obs, double ndof)
{
    // Instantiate the distribution
    boost::math::chi_squared dist(ndof);
    // Returns 1 - CDF efficiently and precisely
    return boost::math::cdf(boost::math::complement(dist, c2obs));
}


int main()
{
    double c2obs = 5.41;
    double ndf = 2.0;

    double pv0 = schi2_test::PValue_0(c2obs, ndf);
    double pv1 = schi2_test::PValue_1(c2obs, ndf);

    std::cout
        << " pv0 " << pv0
        << "\n"
        << " pv1 " << pv1
        << "\n"
        ;

}

