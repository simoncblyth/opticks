#!/usr/bin/env python 

"""





 464 //MT: Lock in caller
 465 void G4SPSEneDistribution::CalculateBbodySpectrum()
 466 {
 467     // create bbody spectrum
 468     // Proved very hard to integrate indefinitely, so different
 469     // method. User inputs emin, emax and T. These are used to
 470     // create a 10,000 bin histogram.
 471     // Use photon density spectrum = 2 nu**2/c**2 * (std::exp(h nu/kT)-1)
 472     // = 2 E**2/h**2c**2 times the exponential
 473 
 474     G4double erange = threadLocalData.Get().Emax - threadLocalData.Get().Emin;
 475     G4double steps = erange / 10000.;
 476 
 477     const G4double k = 8.6181e-11; //Boltzmann const in MeV/K
 478     const G4double h = 4.1362e-21; // Plancks const in MeV s
 479     const G4double c = 3e8; // Speed of light
 480     const G4double h2 = h * h;
 481     const G4double c2 = c * c;
 482     G4int count = 0;
 483     G4double sum = 0.;
 484     BBHist->at(0) = 0.;
 485 
 486     while (count < 10000) {
 487         Bbody_x->at(count) = threadLocalData.Get().Emin + G4double(count * steps);
 488         G4double Bbody_y = (2. * std::pow(Bbody_x->at(count), 2.)) / (h2 * c2 * (std::exp(Bbody_x->at(count) / (k * Temp)) - 1.));
 489         sum = sum + Bbody_y;
 490         BBHist->at(count + 1) = BBHist->at(count) + Bbody_y;
 491         count++;
 492     }
 493 
 494     Bbody_x->at(10000) = threadLocalData.Get().Emax;
 495     // Normalise cumulative histo.
 496     count = 0;
 497     while (count < 10001) {
 498         BBHist->at(count) = BBHist->at(count) / sum;
 499         count++;
 500     }
 501 }


::

    y =         2*x*x
         ---------------------
          h*h*c*c  (exp(x/(k*T)) - 1.)


    cf https://en.wikipedia.org/wiki/Planck%27s_law

    the factor in exponent     hv/(k*T)   so x is hv 


    y  =         2*v*v                             
         --------------------------
           c*c (exp(hv/(kT)) - 1 )


    Which is one v short ? maybe density per unit freq 



"""




