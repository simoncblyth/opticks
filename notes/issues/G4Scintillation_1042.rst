G4Scintillation_1042
=======================


::

    464                 // emission time distribution
    465                 if (ScintillationRiseTime==0.0) {
    466                    deltaTime = deltaTime -
    467                           ScintillationTime * std::log( G4UniformRand() );
    468                 } else {
    469                    deltaTime = deltaTime +
    470                           sample_time(ScintillationRiseTime, ScintillationTime);
    471                 }
    472 
    473                 G4double aSecondaryTime = t0 + deltaTime;


    698 G4double G4Scintillation::sample_time(G4double tau1, G4double tau2)
    699 {
    700 // tau1: rise time and tau2: decay time
    701 
    702     // Loop checking, 07-Aug-2015, Vladimir Ivanchenko
    703         while(1) {
    704           // two random numbers
    705           G4double ran1 = G4UniformRand();
    706           G4double ran2 = G4UniformRand();
    707           //
    708           // exponential distribution as envelope function: very efficient
    709           //
    710           G4double d = (tau1+tau2)/tau2;
    711           // make sure the envelope function is
    712           // always larger than the bi-exponential
    713           G4double t = -1.0*tau2*std::log(1-ran1);
    714           G4double gg = d*single_exp(t,tau2);
    715           if (ran2 <= bi_exp(t,tau1,tau2)/gg) return t;
    716         }
    717         return -1.0;
    718 }

    epsilon:geant4.10.04.p02 blyth$ find source -type f -exec grep -H single_exp {} \;
    source/processes/electromagnetic/xrays/include/G4Scintillation.hh:        G4double single_exp(G4double t, G4double tau2);
    source/processes/electromagnetic/xrays/include/G4Scintillation.hh:G4double G4Scintillation::single_exp(G4double t, G4double tau2)
    source/processes/electromagnetic/xrays/src/G4Scintillation.cc:          G4double gg = d*single_exp(t,tau2);

    epsilon:geant4.10.04.p02 blyth$ find source -type f -exec grep -H bi_exp {} \;
    source/processes/electromagnetic/xrays/include/G4Scintillation.hh:        G4double bi_exp(G4double t, G4double tau1, G4double tau2);
    source/processes/electromagnetic/xrays/include/G4Scintillation.hh:G4double G4Scintillation::bi_exp(G4double t, G4double tau1, G4double tau2)
    source/processes/electromagnetic/xrays/src/G4Scintillation.cc:          if (ran2 <= bi_exp(t,tau1,tau2)/gg) return t;


    394 inline
    395 G4double G4Scintillation::single_exp(G4double t, G4double tau2)
    396 {
    397          return std::exp(-1.0*t/tau2)/tau2;
    398 }   
    399 
    400 inline
    401 G4double G4Scintillation::bi_exp(G4double t, G4double tau1, G4double tau2)
    402 {
    403          return std::exp(-1.0*t/tau2)*(1-std::exp(-1.0*t/tau1))/tau2/tau2*(tau1+tau2);
    404 } 



    epsilon:geant4.10.04.p02 blyth$ find source -type f -exec grep -H G4ScintillationType {} \;
    source/processes/electromagnetic/xrays/include/G4ScintillationTrackInformation.hh:enum G4ScintillationType {Fast, Slow};
    source/processes/electromagnetic/xrays/include/G4ScintillationTrackInformation.hh:    explicit G4ScintillationTrackInformation(const G4ScintillationType& aType = Slow);
    source/processes/electromagnetic/xrays/include/G4ScintillationTrackInformation.hh:    const G4ScintillationType& GetScintillationType() const {return scintillationType;}
    source/processes/electromagnetic/xrays/include/G4ScintillationTrackInformation.hh:    G4ScintillationType scintillationType;
    source/processes/electromagnetic/xrays/src/G4Scintillation.cc:            G4ScintillationType ScintillationType = Slow;
    source/processes/electromagnetic/xrays/src/G4ScintillationTrackInformation.cc:G4ScintillationTrackInformation::G4ScintillationTrackInformation(const G4ScintillationType& aType)
    epsilon:geant4.10.04.p02 blyth$ 


