random_alignment
=====================





skipdupe : Skipping Duplicate Locations
-----------------------------------------

* suppressing duplicate locations, to avoid sampling do/whiles, still leaves
  a handful of unexplained non-unique digest/seqhis relations 


::

    2017-12-06 14:33:31.142 INFO  [512236] [CRandomEngine::dumpDigests@212] CRandomEngine::postpropagate
     total     100000 skipdupe Y
     count      88016 k:digest a95a5c961b4832149e6c00e0b5030091 v:seqhis                             8ccd num_digest_with_seqhis          1
     count       6120 k:digest 1b1df819a447f393d0b43527f3f5f687 v:seqhis                              8bd num_digest_with_seqhis          1
     count       5405 k:digest 58c5ea57f9622b3fb0c8aa8083abe629 v:seqhis                            8cbcd num_digest_with_seqhis          1
     count        319 k:digest 499d2d31d49cc8564d470d967463367b v:seqhis                           8cbbcd num_digest_with_seqhis          1
     count         25 k:digest a7337d2cea87866d40415eb39bffc9b2 v:seqhis                          8cbbbcd num_digest_with_seqhis          1
     count         23 k:digest 8ac3a348be685c9f97d40610090a2569 v:seqhis                            86ccd num_digest_with_seqhis          1
     count         21 k:digest e04ff8d4ab3e7aebf93d4571245e7496 v:seqhis                              86d num_digest_with_seqhis          1
     count         18 k:digest fb9477d53c9da7877c5137576c192bfb v:seqhis                              4cd num_digest_with_seqhis          1
     count          9 k:digest 5db9eaec68e0b85671eb6382f1b9b3dc v:seqhis                       bbbbbbb6cd num_digest_with_seqhis          2
     count          7 k:digest af98026c6adb2c0c86dbaa8d046bf9c5 v:seqhis                            8cc6d num_digest_with_seqhis          1
     count          7 k:digest a36370f0ba4be0496741d55546573dfb v:seqhis                            8c6cd num_digest_with_seqhis          1
     count          6 k:digest a4594d9c1e784f890ff11eb36c7467e1 v:seqhis                               4d num_digest_with_seqhis          1
     count          4 k:digest 3e2458fc062bfa968d894c2233027b55 v:seqhis                             4ccd num_digest_with_seqhis          1
     count          4 k:digest 6ae1c42f08f226e87d52d4ccfdfea9c1 v:seqhis                          8cc6ccd num_digest_with_seqhis          1
     count          2 k:digest 89302d15e1f813009de20699b686dd1c v:seqhis                           86cbcd num_digest_with_seqhis          1
     count          2 k:digest 8807f7cbe9208e84cba42118e4b0d085 v:seqhis                         8cbc6ccd num_digest_with_seqhis          1
     count          1 k:digest 822511c4f603745132253ae9853a1621 v:seqhis                           8cb6cd num_digest_with_seqhis          2
     count          1 k:digest 774eb4a6b992a81ce5066581bfb2b8cf v:seqhis                             86bd num_digest_with_seqhis          1
     count          1 k:digest 679933081af4b8cda6fb216a0b1b058a v:seqhis                       bbbbbbb6cd num_digest_with_seqhis          2
     count          1 k:digest 6301d5f032aab2948b9cfa79e40498d2 v:seqhis                           8cbc6d num_digest_with_seqhis          1
     count          1 k:digest 5069d2b7a3d45c4c2b371499aa96850f v:seqhis                           8cb6cd num_digest_with_seqhis          2
     count          1 k:digest b698f9c0c43c4b335083b684472190cc v:seqhis                         8cbbb6cd num_digest_with_seqhis          1
     count          1 k:digest becc05341fb68e31a2cb58d890d5df0c v:seqhis                        8cbbc6ccd num_digest_with_seqhis          1
     count          1 k:digest bfc7191b635669bf13926430ab5db6ae v:seqhis                          8cbb6cd num_digest_with_seqhis          1
     count          1 k:digest 4e8993d03a0007471f926548290046b5 v:seqhis                           8b6ccd num_digest_with_seqhis          1
     count          1 k:digest e294c3ebb01bba22925cfd880175d115 v:seqhis                            4cbcd num_digest_with_seqhis          1
     count          1 k:digest ef80910914049c0df57b4fa6c54ce927 v:seqhis                         8cbbbc6d num_digest_with_seqhis          1
     count          1 k:digest 055fa8e30937f3a10e808193ab925fa5 v:seqhis                             4bcd num_digest_with_seqhis          1
    2017-12-06 14:33:31.142 INFO  [512236] [CRandomEngine::dumpDigests@212] CRandomEngine::postpropagate
     total     100000 skipdupe Y
     count      88016 k:digest a95a5c961b4832149e6c00e0b5030091 v:seqhis                             8ccd num_digest_with_seqhis          1
     count       6120 k:digest 1b1df819a447f393d0b43527f3f5f687 v:seqhis                              8bd num_digest_with_seqhis          1
     count       5405 k:digest 58c5ea57f9622b3fb0c8aa8083abe629 v:seqhis                            8cbcd num_digest_with_seqhis          1
     count        319 k:digest 499d2d31d49cc8564d470d967463367b v:seqhis                           8cbbcd num_digest_with_seqhis          1
     count         25 k:digest a7337d2cea87866d40415eb39bffc9b2 v:seqhis                          8cbbbcd num_digest_with_seqhis          1
     count         23 k:digest 8ac3a348be685c9f97d40610090a2569 v:seqhis                            86ccd num_digest_with_seqhis          1
     count         21 k:digest e04ff8d4ab3e7aebf93d4571245e7496 v:seqhis                              86d num_digest_with_seqhis          1
     count         18 k:digest fb9477d53c9da7877c5137576c192bfb v:seqhis                              4cd num_digest_with_seqhis          1
     count          9 k:digest 5db9eaec68e0b85671eb6382f1b9b3dc v:seqhis                       bbbbbbb6cd num_digest_with_seqhis          2
    2017-12-06 14:33:31.143 INFO  [512236] [CRandomEngine::dumpLocations@291] dumpLocations ndig 2 nmax 51
                                        Scintillation;                                    Scintillation;
                                           OpBoundary;                                       OpBoundary;
                                           OpRayleigh;                                       OpRayleigh;
                                         OpAbsorption;                                     OpAbsorption;
         OpBoundary;cfg4/DsG4OpBoundaryProcess.cc+1025     OpBoundary;cfg4/DsG4OpBoundaryProcess.cc+1025
                                        Scintillation;                                    Scintillation;
                                           OpBoundary;                                       OpBoundary;
                                           OpRayleigh;                                       OpRayleigh;
                                        Scintillation;                                    Scintillation;
                                           OpBoundary;                                       OpBoundary;
                                           OpRayleigh;                                       OpRayleigh;
                                        Scintillation;                                    Scintillation;
                                           OpBoundary;                                       OpBoundary;
                                        Scintillation;                                    Scintillation;
                                           OpBoundary;                                       OpBoundary;
                                        Scintillation;     OpBoundary;cfg4/DsG4OpBoundaryProcess.cc+1025
                                           OpBoundary;                                    Scintillation;
                                        Scintillation;                                       OpBoundary;
                                           OpBoundary;                                    Scintillation;
                                        Scintillation;                                       OpBoundary;
                                           OpBoundary;                                    Scintillation;
                                        Scintillation;                                       OpBoundary;
                                           OpBoundary;                                    Scintillation;
                                        Scintillation;                                       OpBoundary;
                                           OpBoundary;                                    Scintillation;
                                        Scintillation;                                       OpBoundary;
                                           OpBoundary;                                    Scintillation;
                                        Scintillation;                                       OpBoundary;
                                           OpBoundary;     OpBoundary;cfg4/DsG4OpBoundaryProcess.cc+1025
                                        Scintillation;                                    Scintillation;
                                           OpBoundary;                                       OpBoundary;
                                        Scintillation;                                    Scintillation;
                                           OpBoundary;                                       OpBoundary;
                                        Scintillation;     OpBoundary;cfg4/DsG4OpBoundaryProcess.cc+1025
                                           OpBoundary;                                    Scintillation;
                                        Scintillation;                                       OpBoundary;
                                           OpBoundary;                                    Scintillation;
                                        Scintillation;                                       OpBoundary;
                                           OpBoundary;                                    Scintillation;
                                        Scintillation;                                       OpBoundary;
                                           OpBoundary;                                    Scintillation;
                                        Scintillation;                                       OpBoundary;
                                           OpBoundary;                                    Scintillation;
                                        Scintillation;                                       OpBoundary;
                                           OpBoundary;                                    Scintillation;
                                        Scintillation;                                       OpBoundary;
                                           OpBoundary;     OpBoundary;cfg4/DsG4OpBoundaryProcess.cc+1025
                                        Scintillation;                                    Scintillation;
                                           OpBoundary;                                       OpBoundary;
                                                     -      OpBoundary;cfg4/DsG4OpBoundaryProcess.cc+655
                                                     -       OpBoundary;cfg4/DsG4OpBoundaryProcess.h+269
     count          7 k:digest af98026c6adb2c0c86dbaa8d046bf9c5 v:seqhis                            8cc6d num_digest_with_seqhis          1
     count          7 k:digest a36370f0ba4be0496741d55546573dfb v:seqhis                            8c6cd num_digest_with_seqhis          1
     count          6 k:digest a4594d9c1e784f890ff11eb36c7467e1 v:seqhis                               4d num_digest_with_seqhis          1
     count          4 k:digest 3e2458fc062bfa968d894c2233027b55 v:seqhis                             4ccd num_digest_with_seqhis          1
     count          4 k:digest 6ae1c42f08f226e87d52d4ccfdfea9c1 v:seqhis                          8cc6ccd num_digest_with_seqhis          1
     count          2 k:digest 89302d15e1f813009de20699b686dd1c v:seqhis                           86cbcd num_digest_with_seqhis          1
     count          2 k:digest 8807f7cbe9208e84cba42118e4b0d085 v:seqhis                         8cbc6ccd num_digest_with_seqhis          1
     count          1 k:digest 822511c4f603745132253ae9853a1621 v:seqhis                           8cb6cd num_digest_with_seqhis          2
    2017-12-06 14:33:31.146 INFO  [512236] [CRandomEngine::dumpLocations@291] dumpLocations ndig 2 nmax 21
                                        Scintillation;                                    Scintillation;
                                           OpBoundary;                                       OpBoundary;
                                           OpRayleigh;                                       OpRayleigh;
                                         OpAbsorption;                                     OpAbsorption;
         OpBoundary;cfg4/DsG4OpBoundaryProcess.cc+1025     OpBoundary;cfg4/DsG4OpBoundaryProcess.cc+1025
                                        Scintillation;                                    Scintillation;
                                           OpBoundary;                                       OpBoundary;
                                           OpRayleigh;                                       OpRayleigh;
                                        Scintillation;                                    Scintillation;
                                           OpBoundary;                                       OpBoundary;
                                           OpRayleigh;                                       OpRayleigh;
                                        Scintillation;     OpBoundary;cfg4/DsG4OpBoundaryProcess.cc+1025
                                           OpBoundary;                                    Scintillation;
                                        Scintillation;                                       OpBoundary;
                                           OpBoundary;                                    Scintillation;
         OpBoundary;cfg4/DsG4OpBoundaryProcess.cc+1025                                       OpBoundary;
                                        Scintillation;     OpBoundary;cfg4/DsG4OpBoundaryProcess.cc+1025
                                           OpBoundary;                                    Scintillation;
          OpBoundary;cfg4/DsG4OpBoundaryProcess.cc+655                                       OpBoundary;
           OpBoundary;cfg4/DsG4OpBoundaryProcess.h+269      OpBoundary;cfg4/DsG4OpBoundaryProcess.cc+655
                                                     -       OpBoundary;cfg4/DsG4OpBoundaryProcess.h+269
     count          1 k:digest 774eb4a6b992a81ce5066581bfb2b8cf v:seqhis                             86bd num_digest_with_seqhis          1
     count          1 k:digest 679933081af4b8cda6fb216a0b1b058a v:seqhis                       bbbbbbb6cd num_digest_with_seqhis          2
    2017-12-06 14:33:31.148 INFO  [512236] [CRandomEngine::dumpLocations@291] dumpLocations ndig 2 nmax 51
                                        Scintillation;                                    Scintillation;
                                           OpBoundary;                                       OpBoundary;
                                           OpRayleigh;                                       OpRayleigh;
                                         OpAbsorption;                                     OpAbsorption;
         OpBoundary;cfg4/DsG4OpBoundaryProcess.cc+1025     OpBoundary;cfg4/DsG4OpBoundaryProcess.cc+1025
                                        Scintillation;                                    Scintillation;
                                           OpBoundary;                                       OpBoundary;
                                           OpRayleigh;                                       OpRayleigh;
                                        Scintillation;                                    Scintillation;
                                           OpBoundary;                                       OpBoundary;
                                           OpRayleigh;                                       OpRayleigh;
                                        Scintillation;                                    Scintillation;
                                           OpBoundary;                                       OpBoundary;
                                        Scintillation;                                    Scintillation;
                                           OpBoundary;                                       OpBoundary;
                                        Scintillation;     OpBoundary;cfg4/DsG4OpBoundaryProcess.cc+1025
                                           OpBoundary;                                    Scintillation;
                                        Scintillation;                                       OpBoundary;
                                           OpBoundary;                                    Scintillation;
                                        Scintillation;                                       OpBoundary;
                                           OpBoundary;                                    Scintillation;
                                        Scintillation;                                       OpBoundary;
                                           OpBoundary;                                    Scintillation;
                                        Scintillation;                                       OpBoundary;
                                           OpBoundary;                                    Scintillation;
                                        Scintillation;                                       OpBoundary;
                                           OpBoundary;                                    Scintillation;
                                        Scintillation;                                       OpBoundary;
                                           OpBoundary;     OpBoundary;cfg4/DsG4OpBoundaryProcess.cc+1025
                                        Scintillation;                                    Scintillation;
                                           OpBoundary;                                       OpBoundary;
                                        Scintillation;                                    Scintillation;
                                           OpBoundary;                                       OpBoundary;
                                        Scintillation;     OpBoundary;cfg4/DsG4OpBoundaryProcess.cc+1025
                                           OpBoundary;                                    Scintillation;
                                        Scintillation;                                       OpBoundary;
                                           OpBoundary;                                    Scintillation;
                                        Scintillation;                                       OpBoundary;
                                           OpBoundary;                                    Scintillation;
                                        Scintillation;                                       OpBoundary;
                                           OpBoundary;                                    Scintillation;
                                        Scintillation;                                       OpBoundary;
                                           OpBoundary;                                    Scintillation;
                                        Scintillation;                                       OpBoundary;
                                           OpBoundary;                                    Scintillation;
                                        Scintillation;                                       OpBoundary;
                                           OpBoundary;     OpBoundary;cfg4/DsG4OpBoundaryProcess.cc+1025
                                        Scintillation;                                    Scintillation;
                                           OpBoundary;                                       OpBoundary;
                                                     -      OpBoundary;cfg4/DsG4OpBoundaryProcess.cc+655
                                                     -       OpBoundary;cfg4/DsG4OpBoundaryProcess.h+269
     count          1 k:digest 6301d5f032aab2948b9cfa79e40498d2 v:seqhis                           8cbc6d num_digest_with_seqhis          1
     count          1 k:digest 5069d2b7a3d45c4c2b371499aa96850f v:seqhis                           8cb6cd num_digest_with_seqhis          2
    2017-12-06 14:33:31.151 INFO  [512236] [CRandomEngine::dumpLocations@291] dumpLocations ndig 2 nmax 21
                                        Scintillation;                                    Scintillation;
                                           OpBoundary;                                       OpBoundary;
                                           OpRayleigh;                                       OpRayleigh;
                                         OpAbsorption;                                     OpAbsorption;
         OpBoundary;cfg4/DsG4OpBoundaryProcess.cc+1025     OpBoundary;cfg4/DsG4OpBoundaryProcess.cc+1025
                                        Scintillation;                                    Scintillation;
                                           OpBoundary;                                       OpBoundary;
                                           OpRayleigh;                                       OpRayleigh;
                                        Scintillation;                                    Scintillation;
                                           OpBoundary;                                       OpBoundary;
                                           OpRayleigh;                                       OpRayleigh;
                                        Scintillation;     OpBoundary;cfg4/DsG4OpBoundaryProcess.cc+1025
                                           OpBoundary;                                    Scintillation;
                                        Scintillation;                                       OpBoundary;
                                           OpBoundary;                                    Scintillation;
         OpBoundary;cfg4/DsG4OpBoundaryProcess.cc+1025                                       OpBoundary;
                                        Scintillation;     OpBoundary;cfg4/DsG4OpBoundaryProcess.cc+1025
                                           OpBoundary;                                    Scintillation;
          OpBoundary;cfg4/DsG4OpBoundaryProcess.cc+655                                       OpBoundary;
           OpBoundary;cfg4/DsG4OpBoundaryProcess.h+269      OpBoundary;cfg4/DsG4OpBoundaryProcess.cc+655
                                                     -       OpBoundary;cfg4/DsG4OpBoundaryProcess.h+269
     count          1 k:digest b698f9c0c43c4b335083b684472190cc v:seqhis                         8cbbb6cd num_digest_with_seqhis          1
     count          1 k:digest becc05341fb68e31a2cb58d890d5df0c v:seqhis                        8cbbc6ccd num_digest_with_seqhis          1
     count          1 k:digest bfc7191b635669bf13926430ab5db6ae v:seqhis                          8cbb6cd num_digest_with_seqhis          1
     count          1 k:digest 4e8993d03a0007471f926548290046b5 v:seqhis                           8b6ccd num_digest_with_seqhis          1
     count          1 k:digest e294c3ebb01bba22925cfd880175d115 v:seqhis                            4cbcd num_digest_with_seqhis          1
     count          1 k:digest ef80910914049c0df57b4fa6c54ce927 v:seqhis                         8cbbbc6d num_digest_with_seqhis          1
     count          1 k:digest 055fa8e30937f3a10e808193ab925fa5 v:seqhis                             4bcd num_digest_with_seqhis          1
    2017-12-06 14:33:31.152 INFO  [512236] [CG4::postpropagate@418] CG4::postpropagate(0) DONE






Scattering do/while 
---------------------

::

   g4-;g4-cls G4OpRayleigh



    124 G4OpRayleigh::PostStepDoIt(const G4Track& aTrack, const G4Step& aStep)
    125 {
    126         aParticleChange.Initialize(aTrack);
    127 
    128         const G4DynamicParticle* aParticle = aTrack.GetDynamicParticle();
    129 
    130         if (verboseLevel>0) {
    131                 G4cout << "Scattering Photon!" << G4endl;
    132                 G4cout << "Old Momentum Direction: "
    133                        << aParticle->GetMomentumDirection() << G4endl;
    134                 G4cout << "Old Polarization: "
    135                        << aParticle->GetPolarization() << G4endl;
    136         }
    137 
    138         G4double cosTheta;
    139         G4ThreeVector OldMomentumDirection, NewMomentumDirection;
    140         G4ThreeVector OldPolarization, NewPolarization;
    141 
    142         G4double rand, constant;
    143         G4double CosTheta, SinTheta, SinPhi, CosPhi, unit_x, unit_y, unit_z;
    144 
    145         do {
    146            // Try to simulate the scattered photon momentum direction
    147            // w.r.t. the initial photon momentum direction
    148 
    149            CosTheta = G4UniformRand();
    150            SinTheta = std::sqrt(1.-CosTheta*CosTheta);
    151            // consider for the angle 90-180 degrees
    152            if (G4UniformRand() < 0.5) CosTheta = -CosTheta;
    153 
    154            // simulate the phi angle
    155            rand = twopi*G4UniformRand();
    156            SinPhi = std::sin(rand);
    157            CosPhi = std::cos(rand);
    158 




Relationship between digests (random throw code location sequences) and seqhis
---------------------------------------------------------------------------------

Mostly 1-to-1 but out in the tail some seqhis have multiple digests. 
Dumping these below suggests two causes:

* differing number of random throws in OpRayleigh which doesnt change seqhis
  from the scattering do/while distrib sampling

* truncation handling difference  
  

::

    tboolean-;tboolean-box --okg4 --align 

    2017-12-06 14:01:40.055 INFO  [501997] [CRandomEngine::dumpDigests@205] CRandomEngine::postpropagate
     total     100000
     count      88016 k:digest a95a5c961b4832149e6c00e0b5030091 v:seqhis                             8ccd num_digest_with_seqhis          1
     count       6120 k:digest 1b1df819a447f393d0b43527f3f5f687 v:seqhis                              8bd num_digest_with_seqhis          1
     count       5405 k:digest 58c5ea57f9622b3fb0c8aa8083abe629 v:seqhis                            8cbcd num_digest_with_seqhis          1
     count        319 k:digest 499d2d31d49cc8564d470d967463367b v:seqhis                           8cbbcd num_digest_with_seqhis          1
     count         25 k:digest a7337d2cea87866d40415eb39bffc9b2 v:seqhis                          8cbbbcd num_digest_with_seqhis          1
     count         18 k:digest fb9477d53c9da7877c5137576c192bfb v:seqhis                              4cd num_digest_with_seqhis          1
     count         16 k:digest 274ceb8e0097317bfd3e25c4cc70b714 v:seqhis                            86ccd num_digest_with_seqhis          3
     count         13 k:digest d2c5ac2c3204d033c363ea67c9f71934 v:seqhis                              86d num_digest_with_seqhis          3
     count          7 k:digest 75d211dcf75e64fc68119721fd972e89 v:seqhis                            8cc6d num_digest_with_seqhis          1
     count          6 k:digest a4594d9c1e784f890ff11eb36c7467e1 v:seqhis                               4d num_digest_with_seqhis          1
     count          6 k:digest 36fda1bebbb3148f03eb37d7751a05a2 v:seqhis                              86d num_digest_with_seqhis          3
     count          6 k:digest 2df05b1b610da4a8f4f9ccb326e5e97a v:seqhis                       bbbbbbb6cd num_digest_with_seqhis          4
     count          6 k:digest b59c923c28021d896047521ed92e351c v:seqhis                            86ccd num_digest_with_seqhis          3
     count          4 k:digest 61b45fa4653e261f088663e1eab10121 v:seqhis                            8c6cd num_digest_with_seqhis          3
     count          4 k:digest 3e2458fc062bfa968d894c2233027b55 v:seqhis                             4ccd num_digest_with_seqhis          1
     count          2 k:digest 33483cfc62c24f86ecd7d4479b036026 v:seqhis                       bbbbbbb6cd num_digest_with_seqhis          4
     count          2 k:digest aeebf05b5e147d9f3ecec14b62d57a46 v:seqhis                          8cc6ccd num_digest_with_seqhis          3
     count          2 k:digest 546c458bed524e857ef32599ea3b02d2 v:seqhis                            8c6cd num_digest_with_seqhis          3
     count          2 k:digest 828f79047909333c53b55ffeb97947f6 v:seqhis                              86d num_digest_with_seqhis          3
     count          2 k:digest 6a889a0df5d70e24be3d78f6affd9263 v:seqhis                         8cbc6ccd num_digest_with_seqhis          1
     count          1 k:digest 49921aa3a4c93a94e8a867762137d3cf v:seqhis                          8cc6ccd num_digest_with_seqhis          3
     count          1 k:digest 6e38dc5c540dcd8850e6e4fa678040e8 v:seqhis                           8cb6cd num_digest_with_seqhis          2
     count          1 k:digest 7183b0dcaeb6c60e9a7ec6fa4cc874fb v:seqhis                            86ccd num_digest_with_seqhis          3
     count          1 k:digest 356d7d073f0840e473fcad092bc0d07a v:seqhis                           8cbc6d num_digest_with_seqhis          1
     count          1 k:digest 5c99f339ca26e7957975fcfb08f7c924 v:seqhis                         8cbbb6cd num_digest_with_seqhis          1
     count          1 k:digest 9d4789ce99aba9066bb1d88ec205a97d v:seqhis                           8cb6cd num_digest_with_seqhis          2
     count          1 k:digest 4255041be217d6d098a07eda2a009c2b v:seqhis                        8cbbc6ccd num_digest_with_seqhis          1
     count          1 k:digest 22f82e13b2507b87b0675dace5af55ce v:seqhis                            8c6cd num_digest_with_seqhis          3
     count          1 k:digest 1a19fd38b6311dc7d8db96a8dcf77d23 v:seqhis                          8cbb6cd num_digest_with_seqhis          1
     count          1 k:digest 4a2d1dc415376ad316e8bceecdb288e8 v:seqhis                          8cc6ccd num_digest_with_seqhis          3
     count          1 k:digest 1253705decbaa5ae99781640ba9eab7f v:seqhis                           86cbcd num_digest_with_seqhis          2
     count          1 k:digest ce1da64b26786fcb3d7b4d30c44c4e5c v:seqhis                         8cbbbc6d num_digest_with_seqhis          1
     count          1 k:digest 07e617fd8dcc31dd7b881a82b49af0b9 v:seqhis                             86bd num_digest_with_seqhis          1
     count          1 k:digest e294c3ebb01bba22925cfd880175d115 v:seqhis                            4cbcd num_digest_with_seqhis          1
     count          1 k:digest e2d7bc66afe195c1635c5615105ab831 v:seqhis                       bbbbbbb6cd num_digest_with_seqhis          4
     count          1 k:digest eafd32d3d2a33d5c776b4f128ce58215 v:seqhis                           86cbcd num_digest_with_seqhis          2
     count          1 k:digest eb255d4436a2ac7343e5fa1be471e24a v:seqhis                       bbbbbbb6cd num_digest_with_seqhis          4
     count          1 k:digest ee4213b454f1becfe03d6df2c579fab7 v:seqhis                           8b6ccd num_digest_with_seqhis          1
     count          1 k:digest 055fa8e30937f3a10e808193ab925fa5 v:seqhis                             4bcd num_digest_with_seqhis          1
    2017-12-06 14:01:40.056 INFO  [501997] [CRandomEngine::dumpDigests@205] CRandomEngine::postpropagate
     total     100000
     count      88016 k:digest a95a5c961b4832149e6c00e0b5030091 v:seqhis                             8ccd num_digest_with_seqhis          1
     count       6120 k:digest 1b1df819a447f393d0b43527f3f5f687 v:seqhis                              8bd num_digest_with_seqhis          1
     count       5405 k:digest 58c5ea57f9622b3fb0c8aa8083abe629 v:seqhis                            8cbcd num_digest_with_seqhis          1
     count        319 k:digest 499d2d31d49cc8564d470d967463367b v:seqhis                           8cbbcd num_digest_with_seqhis          1
     count         25 k:digest a7337d2cea87866d40415eb39bffc9b2 v:seqhis                          8cbbbcd num_digest_with_seqhis          1
     count         18 k:digest fb9477d53c9da7877c5137576c192bfb v:seqhis                              4cd num_digest_with_seqhis          1
     count         16 k:digest 274ceb8e0097317bfd3e25c4cc70b714 v:seqhis                            86ccd num_digest_with_seqhis          3
    2017-12-06 14:01:40.057 INFO  [501997] [CRandomEngine::dumpLocations@283] dumpLocations ndig 3 nmax 30
                                        Scintillation;                                    Scintillation;                                    Scintillation;
                                           OpBoundary;                                       OpBoundary;                                       OpBoundary;
                                           OpRayleigh;                                       OpRayleigh;                                       OpRayleigh;
                                         OpAbsorption;                                     OpAbsorption;                                     OpAbsorption;
         OpBoundary;cfg4/DsG4OpBoundaryProcess.cc+1025     OpBoundary;cfg4/DsG4OpBoundaryProcess.cc+1025     OpBoundary;cfg4/DsG4OpBoundaryProcess.cc+1025
                                        Scintillation;                                    Scintillation;                                    Scintillation;
                                           OpBoundary;                                       OpBoundary;                                       OpBoundary;
         OpBoundary;cfg4/DsG4OpBoundaryProcess.cc+1025     OpBoundary;cfg4/DsG4OpBoundaryProcess.cc+1025     OpBoundary;cfg4/DsG4OpBoundaryProcess.cc+1025
                                        Scintillation;                                    Scintillation;                                    Scintillation;
                                           OpBoundary;                                       OpBoundary;                                       OpBoundary;
                                           OpRayleigh;                                       OpRayleigh;                                       OpRayleigh;
                                           OpRayleigh;                                       OpRayleigh;                                       OpRayleigh;
                                           OpRayleigh;                                       OpRayleigh;                                       OpRayleigh;
                                           OpRayleigh;                                       OpRayleigh;                                       OpRayleigh;
                                           OpRayleigh;                                       OpRayleigh;                                       OpRayleigh;
                                        Scintillation;                                       OpRayleigh;                                       OpRayleigh;
                                           OpBoundary;                                       OpRayleigh;                                       OpRayleigh;
                                           OpRayleigh;                                       OpRayleigh;                                       OpRayleigh;
          OpBoundary;cfg4/DsG4OpBoundaryProcess.cc+655                                       OpRayleigh;                                       OpRayleigh;
           OpBoundary;cfg4/DsG4OpBoundaryProcess.h+269                                       OpRayleigh;                                       OpRayleigh;
                                                     -                                       OpRayleigh;                                    Scintillation;
                                                     -                                       OpRayleigh;                                       OpBoundary;
                                                     -                                       OpRayleigh;                                       OpRayleigh;
                                                     -                                       OpRayleigh;      OpBoundary;cfg4/DsG4OpBoundaryProcess.cc+655
                                                     -                                       OpRayleigh;       OpBoundary;cfg4/DsG4OpBoundaryProcess.h+269
                                                     -                                    Scintillation;                                                 -
                                                     -                                       OpBoundary;                                                 -
                                                     -                                       OpRayleigh;                                                 -
                                                     -      OpBoundary;cfg4/DsG4OpBoundaryProcess.cc+655                                                 -
                                                     -       OpBoundary;cfg4/DsG4OpBoundaryProcess.h+269                                                 -
     count         13 k:digest d2c5ac2c3204d033c363ea67c9f71934 v:seqhis                              86d num_digest_with_seqhis          3
    2017-12-06 14:01:40.060 INFO  [501997] [CRandomEngine::dumpLocations@283] dumpLocations ndig 3 nmax 24
                                        Scintillation;                                    Scintillation;                                    Scintillation;
                                           OpBoundary;                                       OpBoundary;                                       OpBoundary;
                                           OpRayleigh;                                       OpRayleigh;                                       OpRayleigh;
                                         OpAbsorption;                                     OpAbsorption;                                     OpAbsorption;
                                           OpRayleigh;                                       OpRayleigh;                                       OpRayleigh;
                                           OpRayleigh;                                       OpRayleigh;                                       OpRayleigh;
                                           OpRayleigh;                                       OpRayleigh;                                       OpRayleigh;
                                           OpRayleigh;                                       OpRayleigh;                                       OpRayleigh;
                                           OpRayleigh;                                       OpRayleigh;                                       OpRayleigh;
                                           OpRayleigh;                                       OpRayleigh;                                    Scintillation;
                                           OpRayleigh;                                       OpRayleigh;                                       OpBoundary;
                                           OpRayleigh;                                       OpRayleigh;                                       OpRayleigh;
                                           OpRayleigh;                                       OpRayleigh;      OpBoundary;cfg4/DsG4OpBoundaryProcess.cc+655
                                           OpRayleigh;                                       OpRayleigh;       OpBoundary;cfg4/DsG4OpBoundaryProcess.h+269
                                        Scintillation;                                       OpRayleigh;                                                 -
                                           OpBoundary;                                       OpRayleigh;                                                 -
                                           OpRayleigh;                                       OpRayleigh;                                                 -
          OpBoundary;cfg4/DsG4OpBoundaryProcess.cc+655                                       OpRayleigh;                                                 -
           OpBoundary;cfg4/DsG4OpBoundaryProcess.h+269                                       OpRayleigh;                                                 -
                                                     -                                    Scintillation;                                                 -
                                                     -                                       OpBoundary;                                                 -
                                                     -                                       OpRayleigh;                                                 -
                                                     -      OpBoundary;cfg4/DsG4OpBoundaryProcess.cc+655                                                 -
                                                     -       OpBoundary;cfg4/DsG4OpBoundaryProcess.h+269                                                 -
     count          7 k:digest 75d211dcf75e64fc68119721fd972e89 v:seqhis                            8cc6d num_digest_with_seqhis          1
     count          6 k:digest a4594d9c1e784f890ff11eb36c7467e1 v:seqhis                               4d num_digest_with_seqhis          1




Observations from CRandomEngine
---------------------------------

* at low stat level, same sequence of code locations for each seqhis


* Q: why does Scintillation and OpBoundary consume a flat at start of every step, 
     but OpRayleigh OpAbsorption consumes only at the first ?




* 31/100k do not have unique relationship between code location vector digest and seqhis


::

    tboolean-;tboolean-box --okg4 --align 

    ...

    2017-12-05 20:42:22.841 ERROR [417523] [CRandomEngine::posttrack@176]  record_id 91063 m_location_vec.size() 31 digest 4a2d1dc415376ad316e8bceecdb288e8 seqhis 8cc6ccd seqmat 1232232 digest/seqhis non-uniqueness  prior 49921aa3a4c93a94e8a867762137d3cf count_mismatch 31
    Scintillation;
    OpBoundary;
    OpRayleigh;
    OpAbsorption;
    OpBoundary;cfg4/DsG4OpBoundaryProcess.cc+1025
    Scintillation;
    OpBoundary;
    OpBoundary;cfg4/DsG4OpBoundaryProcess.cc+1025
    Scintillation;
    OpBoundary;
    OpRayleigh;
    OpRayleigh;
    OpRayleigh;
    OpRayleigh;
    OpRayleigh;
    OpRayleigh;
    OpRayleigh;
    OpRayleigh;
    OpRayleigh;
    OpRayleigh;
    Scintillation;
    OpBoundary;
    OpRayleigh;
    OpBoundary;cfg4/DsG4OpBoundaryProcess.cc+1025
    Scintillation;
    OpBoundary;
    OpBoundary;cfg4/DsG4OpBoundaryProcess.cc+1025
    Scintillation;
    OpBoundary;
    OpBoundary;cfg4/DsG4OpBoundaryProcess.cc+655
    OpBoundary;cfg4/DsG4OpBoundaryProcess.h+269




::


    tboolean-;tboolean-box --okg4 --align --dindex 0:10 --recpoi -D


    2017-12-05 19:43:10.548 INFO  [396009] [CRec::initEvent@82] CRec::initEvent note recpoi not-aligned
    HepRandomEngine::put called -- no effect!
    2017-12-05 19:43:10.844 INFO  [396009] [CRunAction::BeginOfRunAction@19] CRunAction::BeginOfRunAction count 1
     flat   0.286072 record_id     9 count     0 step_id    -1 loc Scintillation;
     flat   0.366332 record_id     9 count     1 step_id    -1 loc OpBoundary;
     flat   0.942989 record_id     9 count     2 step_id    -1 loc OpRayleigh;
     flat   0.278981 record_id     9 count     3 step_id    -1 loc OpAbsorption;
     flat    0.18341 record_id     9 count     4 step_id    -1 loc OpBoundary;cfg4/DsG4OpBoundaryProcess.cc+1025
     flat   0.186724 record_id     9 count     5 step_id     0 loc Scintillation;
     flat   0.265324 record_id     9 count     6 step_id     0 loc OpBoundary;
     flat   0.452413 record_id     9 count     7 step_id     0 loc OpBoundary;cfg4/DsG4OpBoundaryProcess.cc+1025
     flat   0.552432 record_id     9 count     8 step_id     1 loc Scintillation;
     flat   0.223035 record_id     9 count     9 step_id     1 loc OpBoundary;
     flat   0.594206 record_id     9 count    10 step_id     1 loc OpBoundary;cfg4/DsG4OpBoundaryProcess.cc+655
     flat   0.724901 record_id     9 count    11 step_id     1 loc OpBoundary;cfg4/DsG4OpBoundaryProcess.h+269
    2017-12-05 19:43:10.845 INFO  [396009] [CRecorder::posttrack@145] [--dindex]  ctx  record_id 9 pho  seqhis                 8ccd seqmat                 1232
     flat   0.107845 record_id     8 count    12 step_id     2 loc Scintillation;
     flat   0.521342 record_id     8 count    13 step_id     2 loc OpBoundary;
     flat   0.776012 record_id     8 count    14 step_id     2 loc OpRayleigh;
     flat   0.704118 record_id     8 count    15 step_id     2 loc OpAbsorption;
     flat   0.396072 record_id     8 count    16 step_id     2 loc OpBoundary;cfg4/DsG4OpBoundaryProcess.cc+1025
     flat   0.766329 record_id     8 count    17 step_id     0 loc Scintillation;
     flat   0.492083 record_id     8 count    18 step_id     0 loc OpBoundary;
     flat   0.611373 record_id     8 count    19 step_id     0 loc OpBoundary;cfg4/DsG4OpBoundaryProcess.cc+1025
     flat    0.46715 record_id     8 count    20 step_id     1 loc Scintillation;
     flat   0.493843 record_id     8 count    21 step_id     1 loc OpBoundary;
     flat   0.506285 record_id     8 count    22 step_id     1 loc OpBoundary;cfg4/DsG4OpBoundaryProcess.cc+655
     flat   0.230762 record_id     8 count    23 step_id     1 loc OpBoundary;cfg4/DsG4OpBoundaryProcess.h+269
    2017-12-05 19:43:10.846 INFO  [396009] [CRecorder::posttrack@145] [--dindex]  ctx  record_id 8 pho  seqhis                 8ccd seqmat                 1232
     flat   0.786109 record_id     7 count    24 step_id     2 loc Scintillation;
     flat  0.0865933 record_id     7 count    25 step_id     2 loc OpBoundary;
     flat   0.542805 record_id     7 count    26 step_id     2 loc OpRayleigh;
     flat   0.769007 record_id     7 count    27 step_id     2 loc OpAbsorption;
     flat   0.981335 record_id     7 count    28 step_id     2 loc OpBoundary;cfg4/DsG4OpBoundaryProcess.cc+1025
     flat   0.212876 record_id     7 count    29 step_id     0 loc Scintillation;
     flat    0.45667 record_id     7 count    30 step_id     0 loc OpBoundary;
     flat   0.732215 record_id     7 count    31 step_id     1 loc Scintillation;
     flat  0.0547816 record_id     7 count    32 step_id     1 loc OpBoundary;
     flat   0.294668 record_id     7 count    33 step_id     1 loc OpBoundary;cfg4/DsG4OpBoundaryProcess.cc+655
     flat   0.590065 record_id     7 count    34 step_id     1 loc OpBoundary;cfg4/DsG4OpBoundaryProcess.h+269
    2017-12-05 19:43:10.847 INFO  [396009] [CRecorder::posttrack@145] [--dindex]  ctx  record_id 7 pho  seqhis                  8bd seqmat                  122
     flat   0.479438 record_id     6 count    35 step_id     2 loc Scintillation;
     flat   0.734402 record_id     6 count    36 step_id     2 loc OpBoundary;
     flat    0.59692 record_id     6 count    37 step_id     2 loc OpRayleigh;
     flat   0.649783 record_id     6 count    38 step_id     2 loc OpAbsorption;
     flat  0.0815703 record_id     6 count    39 step_id     2 loc OpBoundary;cfg4/DsG4OpBoundaryProcess.cc+1025
     flat   0.588186 record_id     6 count    40 step_id     0 loc Scintillation;
     flat   0.688171 record_id     6 count    41 step_id     0 loc OpBoundary;
     flat   0.968151 record_id     6 count    42 step_id     0 loc OpBoundary;cfg4/DsG4OpBoundaryProcess.cc+1025
     flat   0.510501 record_id     6 count    43 step_id     1 loc Scintillation;
     flat   0.947696 record_id     6 count    44 step_id     1 loc OpBoundary;
     flat   0.492074 record_id     6 count    45 step_id     2 loc Scintillation;
     flat   0.261073 record_id     6 count    46 step_id     2 loc OpBoundary;
     flat   0.813304 record_id     6 count    47 step_id     2 loc OpBoundary;cfg4/DsG4OpBoundaryProcess.cc+1025
     flat   0.338329 record_id     6 count    48 step_id     3 loc Scintillation;
     flat   0.693033 record_id     6 count    49 step_id     3 loc OpBoundary;
     flat   0.660677 record_id     6 count    50 step_id     3 loc OpBoundary;cfg4/DsG4OpBoundaryProcess.cc+655
     flat 0.00901783 record_id     6 count    51 step_id     3 loc OpBoundary;cfg4/DsG4OpBoundaryProcess.h+269
    2017-12-05 19:43:10.848 INFO  [396009] [CRecorder::posttrack@145] [--dindex]  ctx  record_id 6 pho  seqhis                8cbcd seqmat                12332
     flat   0.156998 record_id     5 count    52 step_id     4 loc Scintillation;
     flat    0.34659 record_id     5 count    53 step_id     4 loc OpBoundary;
     flat   0.371647 record_id     5 count    54 step_id     4 loc OpRayleigh;
     flat     0.5632 record_id     5 count    55 step_id     4 loc OpAbsorption;
     flat   0.624632 record_id     5 count    56 step_id     4 loc OpBoundary;cfg4/DsG4OpBoundaryProcess.cc+1025
     flat   0.560517 record_id     5 count    57 step_id     0 loc Scintillation;
     flat   0.999255 record_id     5 count    58 step_id     0 loc OpBoundary;
     flat   0.317415 record_id     5 count    59 step_id     0 loc OpBoundary;cfg4/DsG4OpBoundaryProcess.cc+1025
     flat   0.959877 record_id     5 count    60 step_id     1 loc Scintillation;
     flat   0.356694 record_id     5 count    61 step_id     1 loc OpBoundary;
     flat   0.883787 record_id     5 count    62 step_id     1 loc OpBoundary;cfg4/DsG4OpBoundaryProcess.cc+655
     flat    0.21871 record_id     5 count    63 step_id     1 loc OpBoundary;cfg4/DsG4OpBoundaryProcess.h+269



Harikari/breakpoint locating flat calls
--------------------------------------------

::


    export OPTICKS_CRANDOMENGINE_HARIKARI=0
    tboolean-;tboolean-box --okg4 --align -D


    (lldb) b CRandomEngine::flat 
    Breakpoint 1: no locations (pending).
    WARNING:  Unable to resolve breakpoint to any actual locations.
    (lldb) r


    (lldb) bt
    * thread #1: tid = 0x4c4f7, 0x0000000104478133 libcfg4.dylib`CRandomEngine::flat(this=0x000000010c744a30) + 19 at CRandomEngine.cc:59, queue = 'com.apple.main-thread', stop reason = breakpoint 1.1
        frame #0: 0x0000000104478133 libcfg4.dylib`CRandomEngine::flat(this=0x000000010c744a30) + 19 at CRandomEngine.cc:59
      * frame #1: 0x0000000105ac4b17 libG4processes.dylib`G4VProcess::ResetNumberOfInteractionLengthLeft(this=0x00000001108cffd0) + 23 at G4VProcess.cc:97
        frame #2: 0x0000000105ac6992 libG4processes.dylib`G4VRestDiscreteProcess::PostStepGetPhysicalInteractionLength(this=<unavailable>, track=<unavailable>, previousStepSize=<unavailable>, condition=<unavailable>) + 82 at G4VRestDiscreteProcess.cc:78
        frame #3: 0x0000000105223d67 libG4tracking.dylib`G4SteppingManager::DefinePhysicalStepLength() [inlined] G4VProcess::PostStepGPIL(this=0x00000001108cffd0, track=<unavailable>, previousStepSize=<unavailable>, condition=<unavailable>) + 14 at G4VProcess.hh:503
        frame #4: 0x0000000105223d59 libG4tracking.dylib`G4SteppingManager::DefinePhysicalStepLength(this=0x0000000110850420) + 249 at G4SteppingManager2.cc:172
        frame #5: 0x000000010522273e libG4tracking.dylib`G4SteppingManager::Stepping(this=0x0000000110850420) + 366 at G4SteppingManager.cc:180
        frame #6: 0x000000010522c771 libG4tracking.dylib`G4TrackingManager::ProcessOneTrack(this=0x00000001108503e0, apValueG4Track=<unavailable>) + 913 at G4TrackingManager.cc:126
        frame #7: 0x0000000105184727 libG4event.dylib`G4EventManager::DoProcessing(this=0x0000000110850350, anEvent=<unavailable>) + 1879 at G4EventManager.cc:185
        frame #8: 0x0000000105106611 libG4run.dylib`G4RunManager::ProcessOneEvent(this=0x000000010c744ee0, i_event=0) + 49 at G4RunManager.cc:399
        frame #9: 0x00000001051064db libG4run.dylib`G4RunManager::DoEventLoop(this=0x000000010c744ee0, n_event=1, macroFile=<unavailable>, n_select=<unavailable>) + 43 at G4RunManager.cc:367
        frame #10: 0x0000000105105913 libG4run.dylib`G4RunManager::BeamOn(this=0x000000010c744ee0, n_event=1, macroFile=0x0000000000000000, n_select=-1) + 99 at G4RunManager.cc:273
        frame #11: 0x0000000104473fc6 libcfg4.dylib`CG4::propagate(this=0x000000010c744840) + 1670 at CG4.cc:354
        frame #12: 0x000000010457b25a libokg4.dylib`OKG4Mgr::propagate(this=0x00007fff5fbfdec0) + 538 at OKG4Mgr.cc:88
        frame #13: 0x00000001000132da OKG4Test`main(argc=30, argv=0x00007fff5fbfdfa0) + 1498 at OKG4Test.cc:57
        frame #14: 0x00007fff8b7125fd libdyld.dylib`start + 1
    (lldb) f 1
    frame #1: 0x0000000105ac4b17 libG4processes.dylib`G4VProcess::ResetNumberOfInteractionLengthLeft(this=0x00000001108cffd0) + 23 at G4VProcess.cc:97
       94   
       95   void G4VProcess::ResetNumberOfInteractionLengthLeft()
       96   {
    -> 97     theNumberOfInteractionLengthLeft =  -std::log( G4UniformRand() );
       98     theInitialNumberOfInteractionLength = theNumberOfInteractionLengthLeft; 
       99   }
       100 
      

    ##  1st 4 consumptions all from same piece of code for each active process
    ##
    ##   (i think the first 2 of these are never used, they have no Opticks equivalent )
    ##    will need to artificially burn these two to stay aligned 

    (lldb) p this->theProcessName
    (G4String) $0 = (std::__1::string = "Scintillation")

    (lldb) p this->theProcessName
    (G4String) $1 = (std::__1::string = "OpBoundary")


    ## the below two have direct equivalent

    (lldb) p this->theProcessName
    (G4String) $2 = (std::__1::string = "OpRayleigh")
    (lldb) 
    (lldb) p this->theProcessName
    (G4String) $3 = (std::__1::string = "OpAbsorption")
    (lldb) 


Direct equivalents::

     59 __device__ int propagate_to_boundary( Photon& p, State& s, curandState &rng)
     60 {
     61     //float speed = SPEED_OF_LIGHT/s.material1.x ;    // .x:refractive_index    (phase velocity of light in medium)
     62     float speed = s.m1group2.x ;  // .x:group_velocity  (group velocity of light in the material) see: opticks-find GROUPVEL
     63 
     64     float absorption_distance = -s.material1.y*logf(curand_uniform(&rng));   // .y:absorption_length
     65     float scattering_distance = -s.material1.z*logf(curand_uniform(&rng));   // .z:scattering_length
     66 


    (lldb) bt
    * thread #1: tid = 0x4c4f7, 0x0000000104478133 libcfg4.dylib`CRandomEngine::flat(this=0x000000010c744a30) + 19 at CRandomEngine.cc:59, queue = 'com.apple.main-thread', stop reason = breakpoint 1.1
      * frame #0: 0x0000000104478133 libcfg4.dylib`CRandomEngine::flat(this=0x000000010c744a30) + 19 at CRandomEngine.cc:59
        frame #1: 0x000000010439875d libcfg4.dylib`DsG4OpBoundaryProcess::PostStepDoIt(this=0x00000001108d1ea0, aTrack=0x000000011eaef750, aStep=0x00000001108505b0) + 7357 at DsG4OpBoundaryProcess.cc:655
        frame #2: 0x0000000105224e2b libG4tracking.dylib`G4SteppingManager::InvokePSDIP(this=0x0000000110850420, np=<unavailable>) + 59 at G4SteppingManager2.cc:530
        frame #3: 0x0000000105224d2b libG4tracking.dylib`G4SteppingManager::InvokePostStepDoItProcs(this=0x0000000110850420) + 139 at G4SteppingManager2.cc:502
        frame #4: 0x0000000105222909 libG4tracking.dylib`G4SteppingManager::Stepping(this=0x0000000110850420) + 825 at G4SteppingManager.cc:209
        frame #5: 0x000000010522c771 libG4tracking.dylib`G4TrackingManager::ProcessOneTrack(this=0x00000001108503e0, apValueG4Track=<unavailable>) + 913 at G4TrackingManager.cc:126
        frame #6: 0x0000000105184727 libG4event.dylib`G4EventManager::DoProcessing(this=0x0000000110850350, anEvent=<unavailable>) + 1879 at G4EventManager.cc:185
        frame #7: 0x0000000105106611 libG4run.dylib`G4RunManager::ProcessOneEvent(this=0x000000010c744ee0, i_event=0) + 49 at G4RunManager.cc:399
        frame #8: 0x00000001051064db libG4run.dylib`G4RunManager::DoEventLoop(this=0x000000010c744ee0, n_event=1, macroFile=<unavailable>, n_select=<unavailable>) + 43 at G4RunManager.cc:367
        frame #9: 0x0000000105105913 libG4run.dylib`G4RunManager::BeamOn(this=0x000000010c744ee0, n_event=1, macroFile=0x0000000000000000, n_select=-1) + 99 at G4RunManager.cc:273
        frame #10: 0x0000000104473fc6 libcfg4.dylib`CG4::propagate(this=0x000000010c744840) + 1670 at CG4.cc:354
        frame #11: 0x000000010457b25a libokg4.dylib`OKG4Mgr::propagate(this=0x00007fff5fbfdec0) + 538 at OKG4Mgr.cc:88
        frame #12: 0x00000001000132da OKG4Test`main(argc=30, argv=0x00007fff5fbfdfa0) + 1498 at OKG4Test.cc:57
        frame #13: 0x00007fff8b7125fd libdyld.dylib`start + 1
    (lldb) 
    (lldb) f 1
    frame #1: 0x000000010439875d libcfg4.dylib`DsG4OpBoundaryProcess::PostStepDoIt(this=0x00000001108d1ea0, aTrack=0x000000011eaef750, aStep=0x00000001108505b0) + 7357 at DsG4OpBoundaryProcess.cc:655
       652  
       653  
       654  #ifdef SCB_REFLECT_CHEAT
    -> 655                  G4double _u = m_reflectcheat ? m_g4->getCtxRecordFraction()  : G4UniformRand() ;   // --reflectcheat 
       656                  bool _reflect = _u < theReflectivity ;
       657                  if( !_reflect ) 
       658  #else
    (lldb) p m_reflectcheat
    (bool) $4 = false
    (lldb) 


::

    (lldb) bt
    * thread #1: tid = 0x4c4f7, 0x0000000104478133 libcfg4.dylib`CRandomEngine::flat(this=0x000000010c744a30) + 19 at CRandomEngine.cc:59, queue = 'com.apple.main-thread', stop reason = breakpoint 1.1
      * frame #0: 0x0000000104478133 libcfg4.dylib`CRandomEngine::flat(this=0x000000010c744a30) + 19 at CRandomEngine.cc:59
        frame #1: 0x000000010439e6d7 libcfg4.dylib`DsG4OpBoundaryProcess::G4BooleanRand(this=0x00000001108d1ea0, prob=0) const + 39 at DsG4OpBoundaryProcess.h:264
        frame #2: 0x000000010439baeb libcfg4.dylib`DsG4OpBoundaryProcess::DoAbsorption(this=0x00000001108d1ea0) + 43 at DsG4OpBoundaryProcess.cc:1240
        frame #3: 0x00000001043987cf libcfg4.dylib`DsG4OpBoundaryProcess::PostStepDoIt(this=0x00000001108d1ea0, aTrack=0x000000011eaef750, aStep=0x00000001108505b0) + 7471 at DsG4OpBoundaryProcess.cc:662
        frame #4: 0x0000000105224e2b libG4tracking.dylib`G4SteppingManager::InvokePSDIP(this=0x0000000110850420, np=<unavailable>) + 59 at G4SteppingManager2.cc:530
        frame #5: 0x0000000105224d2b libG4tracking.dylib`G4SteppingManager::InvokePostStepDoItProcs(this=0x0000000110850420) + 139 at G4SteppingManager2.cc:502
        frame #6: 0x0000000105222909 libG4tracking.dylib`G4SteppingManager::Stepping(this=0x0000000110850420) + 825 at G4SteppingManager.cc:209
        frame #7: 0x000000010522c771 libG4tracking.dylib`G4TrackingManager::ProcessOneTrack(this=0x00000001108503e0, apValueG4Track=<unavailable>) + 913 at G4TrackingManager.cc:126
        frame #8: 0x0000000105184727 libG4event.dylib`G4EventManager::DoProcessing(this=0x0000000110850350, anEvent=<unavailable>) + 1879 at G4EventManager.cc:185
        frame #9: 0x0000000105106611 libG4run.dylib`G4RunManager::ProcessOneEvent(this=0x000000010c744ee0, i_event=0) + 49 at G4RunManager.cc:399
        frame #10: 0x00000001051064db libG4run.dylib`G4RunManager::DoEventLoop(this=0x000000010c744ee0, n_event=1, macroFile=<unavailable>, n_select=<unavailable>) + 43 at G4RunManager.cc:367
        frame #11: 0x0000000105105913 libG4run.dylib`G4RunManager::BeamOn(this=0x000000010c744ee0, n_event=1, macroFile=0x0000000000000000, n_select=-1) + 99 at G4RunManager.cc:273
        frame #12: 0x0000000104473fc6 libcfg4.dylib`CG4::propagate(this=0x000000010c744840) + 1670 at CG4.cc:354
        frame #13: 0x000000010457b25a libokg4.dylib`OKG4Mgr::propagate(this=0x00007fff5fbfdec0) + 538 at OKG4Mgr.cc:88
        frame #14: 0x00000001000132da OKG4Test`main(argc=30, argv=0x00007fff5fbfdfa0) + 1498 at OKG4Test.cc:57
        frame #15: 0x00007fff8b7125fd libdyld.dylib`start + 1
    (lldb) 

    (lldb) f 2
    frame #2: 0x000000010439baeb libcfg4.dylib`DsG4OpBoundaryProcess::DoAbsorption(this=0x00000001108d1ea0) + 43 at DsG4OpBoundaryProcess.cc:1240
       1237 
       1238     theStatus = Absorption;
       1239 
    -> 1240     if ( G4BooleanRand(theEfficiency) ) 
       1241     {
       1242         // EnergyDeposited =/= 0 means: photon has been detected
       1243         theStatus = Detection;
    (lldb) p this->theProcessName
    (G4String) $6 = (std::__1::string = "OpBoundary")





TO_SA[4]::

     648         else if (type == dielectric_dielectric)
     649         {
     650             if ( theFinish == polishedfrontpainted || theFinish == groundfrontpainted )
     651             {
     652 
     653 
     654 #ifdef SCB_REFLECT_CHEAT
     655                 G4double _u = m_reflectcheat ? m_g4->getCtxRecordFraction()  : G4UniformRand() ;   // --reflectcheat 
     656                 bool _reflect = _u < theReflectivity ;
     657                 if( !_reflect )
     658 #else
     659                 if( !G4BooleanRand(theReflectivity) )
     660 #endif
     661                 {
     662                     DoAbsorption();
     663                 }
     664                 else
     665                 {
     666                     if ( theFinish == groundfrontpainted ) theStatus = LambertianReflection;
     667                     DoReflection();
     668                 }
     669             }
     670             else


TO_SA[5]::

    1232 void DsG4OpBoundaryProcess::DoAbsorption()
    1233 {
    1234     //LOG(info) << "DsG4OpBoundaryProcess::DoAbsorption"
    1235     //          << " theEfficiency " << theEfficiency
    1236     //          ; 
    1237 
    1238     theStatus = Absorption;
    1239 
    1240     if ( G4BooleanRand(theEfficiency) )
    1241     {
    1242         // EnergyDeposited =/= 0 means: photon has been detected
    1243         theStatus = Detection;
    1244         aParticleChange.ProposeLocalEnergyDeposit(thePhotonMomentum);
    1245     }
    1246     else
    1247     {
    1248         aParticleChange.ProposeLocalEnergyDeposit(0.0);
    1249     }
    1250 
    1251     NewMomentum = OldMomentum;
    1252     NewPolarization = OldPolarization;
    1253 
    1254 //  aParticleChange.ProposeEnergy(0.0);
    1255     aParticleChange.ProposeTrackStatus(fStopAndKill);
    1256 }





* TO SA 


====  =================   =====================================================   =====================================================================
gen    proc                 loc
====  =================   =====================================================   =====================================================================
 0     Scintillation        G4VProcess::ResetNumberOfInteractionLengthLeft          theNumberOfInteractionLengthLeft =  -std::log( G4UniformRand() );
 1     OpBoundary           G4VProcess::ResetNumberOfInteractionLengthLeft          ditto 
 2     OpRayleigh           G4VProcess::ResetNumberOfInteractionLengthLeft          ditto
 3     OpAbsorption         G4VProcess::ResetNumberOfInteractionLengthLeft          ditto
----  -----------------   -----------------------------------------------------   ---------------------------------------------------------------------
 4     OpBoundary           DsG4OpBoundaryProcess::PostStepDoIt                     theReflectivity decision (+655)
 5     OpBoundary           DsG4OpBoundaryProcess::DoAbsorption                     theEfficiency decision (+1240)   
====  =================   =====================================================   =====================================================================





::

    (lldb) bt
    * thread #1: tid = 0x4e7a4, 0x0000000104478133 libcfg4.dylib`CRandomEngine::flat(this=0x000000010f4029a0) + 19 at CRandomEngine.cc:59, queue = 'com.apple.main-thread', stop reason = breakpoint 1.1
      * frame #0: 0x0000000104478133 libcfg4.dylib`CRandomEngine::flat(this=0x000000010f4029a0) + 19 at CRandomEngine.cc:59
        frame #1: 0x000000010439c822 libcfg4.dylib`DsG4OpBoundaryProcess::DielectricDielectric(this=0x000000011008fe90) + 3202 at DsG4OpBoundaryProcess.cc:1025
        frame #2: 0x0000000104398828 libcfg4.dylib`DsG4OpBoundaryProcess::PostStepDoIt(this=0x000000011008fe90, aTrack=0x000000011e804640, aStep=0x000000011000e560) + 7560 at DsG4OpBoundaryProcess.cc:672
        frame #3: 0x0000000105224e2b libG4tracking.dylib`G4SteppingManager::InvokePSDIP(this=0x000000011000e3d0, np=<unavailable>) + 59 at G4SteppingManager2.cc:530
        frame #4: 0x0000000105224d2b libG4tracking.dylib`G4SteppingManager::InvokePostStepDoItProcs(this=0x000000011000e3d0) + 139 at G4SteppingManager2.cc:502
        frame #5: 0x0000000105222909 libG4tracking.dylib`G4SteppingManager::Stepping(this=0x000000011000e3d0) + 825 at G4SteppingManager.cc:209
        frame #6: 0x000000010522c771 libG4tracking.dylib`G4TrackingManager::ProcessOneTrack(this=0x000000011000e390, apValueG4Track=<unavailable>) + 913 at G4TrackingManager.cc:126
        frame #7: 0x0000000105184727 libG4event.dylib`G4EventManager::DoProcessing(this=0x000000011000e300, anEvent=<unavailable>) + 1879 at G4EventManager.cc:185
        frame #8: 0x0000000105106611 libG4run.dylib`G4RunManager::ProcessOneEvent(this=0x000000010f402e50, i_event=0) + 49 at G4RunManager.cc:399
        frame #9: 0x00000001051064db libG4run.dylib`G4RunManager::DoEventLoop(this=0x000000010f402e50, n_event=1, macroFile=<unavailable>, n_select=<unavailable>) + 43 at G4RunManager.cc:367
        frame #10: 0x0000000105105913 libG4run.dylib`G4RunManager::BeamOn(this=0x000000010f402e50, n_event=1, macroFile=0x0000000000000000, n_select=-1) + 99 at G4RunManager.cc:273
        frame #11: 0x0000000104473fc6 libcfg4.dylib`CG4::propagate(this=0x000000010f4027b0) + 1670 at CG4.cc:354
        frame #12: 0x000000010457b25a libokg4.dylib`OKG4Mgr::propagate(this=0x00007fff5fbfdec0) + 538 at OKG4Mgr.cc:88
        frame #13: 0x00000001000132da OKG4Test`main(argc=30, argv=0x00007fff5fbfdfa0) + 1498 at OKG4Test.cc:57
        frame #14: 0x00007fff8b7125fd libdyld.dylib`start + 1
    (lldb) f 1
    frame #1: 0x000000010439c822 libcfg4.dylib`DsG4OpBoundaryProcess::DielectricDielectric(this=0x000000011008fe90) + 3202 at DsG4OpBoundaryProcess.cc:1025
       1022           G4double E2_abs, C_parl, C_perp;
       1023 
       1024 #ifdef SCB_REFLECT_CHEAT 
    -> 1025           G4double _u = m_reflectcheat ? m_g4->getCtxRecordFraction()  : G4UniformRand() ;   // --reflectcheat 
       1026           bool _transmit = _u < TransCoeff ; 
       1027           if ( !_transmit ) {
       1028 #else
    (lldb) 




Repeating with "TO 


DsG4OpBoundaryProcess::DielectricDielectric  DsG4OpBoundaryProcess.cc +1025

::

    1022           G4double E2_abs, C_parl, C_perp;
    1023 
    1024 #ifdef SCB_REFLECT_CHEAT 
    1025           G4double _u = m_reflectcheat ? m_g4->getCtxRecordFraction()  : G4UniformRand() ;   // --reflectcheat 
    1026           bool _transmit = _u < TransCoeff ;
    1027           if ( !_transmit ) {
    1028 #else
    1029           if ( !G4BooleanRand(TransCoeff) ) {
    1030 #endif
    1031 
    1032              // Simulate reflection
    1033 


::

    (lldb) p theProcessName
    (G4String) $4 = (std::__1::string = "Scintillation")



Hmm need to access current process, so can dump a summary 
-------------------------------------------------------------

::

    (lldb) p theProcessName
    (G4String) $6 = (std::__1::string = "OpBoundary")
    (lldb) f 3
    frame #3: 0x0000000105224e2b libG4tracking.dylib`G4SteppingManager::InvokePSDIP(this=0x000000011000e3d0, np=<unavailable>) + 59 at G4SteppingManager2.cc:530
       527  {
       528           fCurrentProcess = (*fPostStepDoItVector)[np];
       529           fParticleChange 
    -> 530              = fCurrentProcess->PostStepDoIt( *fTrack, *fStep);
       531  
       532           // Update PostStepPoint of Step according to ParticleChange
       533       fParticleChange->UpdateStepForPostStep(fStep);
    (lldb) 




TO BT BT SA::

    2017-12-05 16:31:24.775 INFO  [339989] [CRec::initEvent@82] CRec::initEvent note recstp
    HepRandomEngine::put called -- no effect!
    2017-12-05 16:31:25.073 INFO  [339989] [CRunAction::BeginOfRunAction@19] CRunAction::BeginOfRunAction count 1

    2017-12-05 16:31:25.074 INFO  [339989] [CRandomEngine::flat@71]  record_id 0 count 0 flat 0.286072 processName Scintillation
    2017-12-05 16:31:25.074 INFO  [339989] [CRandomEngine::flat@71]  record_id 0 count 1 flat 0.366332 processName OpBoundary
    2017-12-05 16:31:25.074 INFO  [339989] [CRandomEngine::flat@71]  record_id 0 count 2 flat 0.942989 processName OpRayleigh
    2017-12-05 16:31:25.074 INFO  [339989] [CRandomEngine::flat@71]  record_id 0 count 3 flat 0.278981 processName OpAbsorption

    2017-12-05 16:31:25.074 INFO  [339989] [CRandomEngine::flat@71]  record_id 0 count 4 flat 0.18341 processName OpBoundary

    2017-12-05 16:31:25.074 INFO  [339989] [CRandomEngine::flat@71]  record_id 0 count 5 flat 0.186724 processName Scintillation
    2017-12-05 16:31:25.074 INFO  [339989] [CRandomEngine::flat@71]  record_id 0 count 6 flat 0.265324 processName OpBoundary
    2017-12-05 16:31:25.074 INFO  [339989] [CRandomEngine::flat@71]  record_id 0 count 7 flat 0.452413 processName OpBoundary

    2017-12-05 16:31:25.074 INFO  [339989] [CRandomEngine::flat@71]  record_id 0 count 8 flat 0.552432 processName Scintillation
    2017-12-05 16:31:25.074 INFO  [339989] [CRandomEngine::flat@71]  record_id 0 count 9 flat 0.223035 processName OpBoundary
    2017-12-05 16:31:25.074 INFO  [339989] [CRandomEngine::flat@71]  record_id 0 count 10 flat 0.594206 processName OpBoundary
    2017-12-05 16:31:25.074 INFO  [339989] [CRandomEngine::flat@71]  record_id 0 count 11 flat 0.724901 processName OpBoundary
    2017-12-05 16:31:25.074 INFO  [339989] [CRunAction::EndOfRunAction@23] CRunAction::EndOfRunAction count 1






CRandomEngine standin to investigate number and position of G4UniformRand flat calls
---------------------------------------------------------------------------------------

::

    tboolean-;tboolean-box --okg4 --align -D

::

    2017-12-04 21:02:54.323 INFO  [208401] [CGenerator::configureEvent@124] CGenerator:configureEvent fabricated TORCH genstep (STATIC RUNNING) 
    2017-12-04 21:02:54.323 INFO  [208401] [CG4Ctx::initEvent@134] CG4Ctx::initEvent photons_per_g4event 10000 steps_per_photon 10 gen 4096
    2017-12-04 21:02:54.323 INFO  [208401] [CWriter::initEvent@69] CWriter::initEvent dynamic STATIC(GPU style) record_max 1 bounce_max  9 steps_per_photon 10 num_g4event 1
    2017-12-04 21:02:54.323 INFO  [208401] [CRec::initEvent@82] CRec::initEvent note recstp
    HepRandomEngine::put called -- no effect!
    2017-12-04 21:02:54.629 INFO  [208401] [CRunAction::BeginOfRunAction@19] CRunAction::BeginOfRunAction count 1
    2017-12-04 21:02:54.631 INFO  [208401] [CRandomEngine::flat@56]  record_id 0 count 0
    2017-12-04 21:02:54.631 INFO  [208401] [CRandomEngine::flat@56]  record_id 0 count 1
    2017-12-04 21:02:54.631 INFO  [208401] [CRandomEngine::flat@56]  record_id 0 count 2
    2017-12-04 21:02:54.631 INFO  [208401] [CRandomEngine::flat@56]  record_id 0 count 3
    2017-12-04 21:02:54.631 INFO  [208401] [CRandomEngine::flat@56]  record_id 0 count 4
    2017-12-04 21:02:54.631 INFO  [208401] [CRandomEngine::flat@56]  record_id 0 count 5
    2017-12-04 21:02:54.631 INFO  [208401] [CRunAction::EndOfRunAction@23] CRunAction::EndOfRunAction count 1
    2017-12-04 21:02:54.632 INFO  [208401] [CG4::postpropagate@373] CG4::postpropagate(0) ctx CG4Ctx::desc_stats dump_count 0 event_total 1 event_track_count 1
    2017-12-04 21:02:54.632 INFO  [208401] [OpticksEvent::postPropagateGeant4@2040] OpticksEvent::postPropagateGeant4 shape  genstep 1,6,4 nopstep 0,4,4 photon 1,4,4 source 1,4,4 record 1,10,2,4 phosel 1,1,4 recsel 1,10,1,4 sequence 1,1,2 seed 1,1,1 hit 0,4,4 num_photons 1
    2017-12-04 21:02:54.632 INFO  [208401] [OpticksEvent::indexPhotonsCPU@2086] OpticksEvent::indexPhotonsCPU sequence 1,1,2 phosel 1,1,4 phosel.hasData 0 recsel0 1,10,1,4 recsel0.hasData 0
    2017-12-04 21:02:54.632 INFO  [208401] [OpticksEvent::indexPhotonsCPU@2103] indexSequence START 



::

    simon:opticks blyth$ g4-cc HepRandomEngine::put
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/externals/clhep/src/RandomEngine.cc:std::ostream & HepRandomEngine::put (std::ostream & os) const {
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/externals/clhep/src/RandomEngine.cc:  std::cerr << "HepRandomEngine::put called -- no effect!\n";
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/externals/clhep/src/RandomEngine.cc:std::vector<unsigned long> HepRandomEngine::put () const {
    /usr/local/opticks/externals/g4/geant4_10_02_p01/source/externals/clhep/src/RandomEngine.cc:  std::cerr << "v=HepRandomEngine::put() called -- no data!\n";
    simon:opticks blyth$ vi /usr/local/opticks/externals/g4/geant4_10_02_p01/source/externals/clhep/src/RandomEngine.cc
    simon:opticks blyth$ 



difficult step : aligning consumption
----------------------------------------

Arrange Opticks code random consumption sequence matches the G4 one
  
This would clearly be impossible(prohibitively expensive to do) 
in a general simulation, but in simulations restricted to 
optical photons (using a subselection of materials/surfaces etc..) 
with initial photons provided as input it seems like it may be possible.

tractable
-----------

get curand to duplicate the GPU sequence on the host, such 
that have random access to per photon slot sequences : these being 
used at startTrack level to feed the sequence into NonRandomEngine

Could just grab from GPU, but would entail wasting a lot of space
as would need to get for every photon the maximum sequence length
that the bounciest truncated photon required.

::

    cudarap/tests/curand_aligned_device.cu
    cudarap/tests/curand_aligned_host.c

* currently the curand host api only working up to slot 4095
* but can just use thrust to random access any slots sequence


TRngBufTest
------------

Produces using curand/thrust 16 floats per photon slot(just example number), 
reproducing the generate.cu in-situ ones from --zrngtest 

Initially attemping to generate 1M at once, got resource issues,
so split the thrust "launches".

::

    simon:tests blyth$ TRngBufTest 
    2017-12-02 20:04:12.284 INFO  [21910] [main@21] TRngBufTest
    TRngBuf::generate ni 100000 id_max 1000
    TRngBuf::generate seq 0 id_offset          0 id_per_gen       1000 remaining     100000
    TRngBuf::generate seq 1 id_offset       1000 id_per_gen       1000 remaining      99000
    TRngBuf::generate seq 2 id_offset       2000 id_per_gen       1000 remaining      98000
    ...
    TRngBuf::generate seq 96 id_offset      96000 id_per_gen       1000 remaining       4000
    TRngBuf::generate seq 97 id_offset      97000 id_per_gen       1000 remaining       3000
    TRngBuf::generate seq 98 id_offset      98000 id_per_gen       1000 remaining       2000
    TRngBuf::generate seq 99 id_offset      99000 id_per_gen       1000 remaining       1000
    (100000, 4, 4)
    [[[ 0.74021935  0.43845114  0.51701266  0.15698862]
      [ 0.07136751  0.46250838  0.22764327  0.32935849]
      [ 0.14406531  0.18779911  0.91538346  0.54012483]
      [ 0.97466087  0.54746926  0.65316027  0.23023781]]

     [[ 0.9209938   0.46036443  0.33346406  0.37252042]
      [ 0.48960248  0.56727093  0.07990581  0.23336816]
      [ 0.50937784  0.08897854  0.00670976  0.95422709]
      [ 0.54671133  0.82454693  0.52706289  0.93013161]]

     [[ 0.03902049  0.25021473  0.18448432  0.96242225]
      [ 0.5205546   0.93996495  0.83057821  0.40973285]
      [ 0.08162197  0.80677092  0.69528568  0.61770737]
      [ 0.25633496  0.21368156  0.34242383  0.22407883]]

     ..., 
     [[ 0.81814659  0.20170131  0.54593664  0.04129851]
      [ 0.38002208  0.91853744  0.02320537  0.05250723]
      [ 0.11425403  0.77515221  0.40338024  0.97540855]
      [ 0.46321765  0.80014837  0.65215546  0.73192346]]

     [[ 0.62748933  0.05319326  0.34443355  0.8561672 ]
      [ 0.2001164   0.3857657   0.31989732  0.40597615]
      [ 0.45497316  0.97913557  0.64739084  0.81499505]
      [ 0.82874513  0.009322    0.81717068  0.57686758]]

     [[ 0.91401154  0.44032493  0.94783556  0.09001808]
      [ 0.9587481   0.98795038  0.2274524   0.04384946]
      [ 0.77744925  0.50308371  0.30509573  0.18650141]
      [ 0.32255048  0.73956126  0.63323611  0.65263885]]]
    simon:tests blyth$ 

::

    In [1]: import os, numpy as np

    In [2]: c = np.load(os.path.expandvars("$TMP/TRngBufTest.npy"))

    In [3]: a = np.load("/tmp/blyth/opticks/evt/tboolean-box/torch/1/ox.npy")

    In [4]: np.all( a == c )
    Out[4]: True



curand aligned with G4 random ?
------------------------------------

Suspect getting different imps of generators
to provide same sequences, would be an exercise in frustration.
And in any case the way curand works, having a "cursor" for each 
photon slot to allow parallel usage means that need to 
operate slot-by-slot.
  
But Geant4 has a NonRandomEngine, which enables
the sequence to be provided as input, see cfg4/tests/G4UniformRandTest.cc 

* reemission would be a complication, because its done all in one go
  with Opticks but in two(or more) separate tracks with Geant4


review G4 random
------------------

::

   g4-;g4-cls Randomize
   g4-;g4-cls Random
   g4-;g4-cls RandomEngine
   g4-;g4-cls NonRandomEngine
   g4-;g4-cls JamesRandom


::

    simon:Random blyth$ grep public\ HepRandomEngine *.*

    DualRand.h:class DualRand: public HepRandomEngine {
    JamesRandom.h:class HepJamesRandom: public HepRandomEngine {
    MTwistEngine.h:class MTwistEngine : public HepRandomEngine {
    MixMaxRng.h:class MixMaxRng: public HepRandomEngine {
    NonRandomEngine.h:class NonRandomEngine : public HepRandomEngine {
    RanecuEngine.h:class RanecuEngine : public HepRandomEngine {
    Ranlux64Engine.h:class Ranlux64Engine : public HepRandomEngine {
    RanluxEngine.h:class RanluxEngine : public HepRandomEngine {
    RanshiEngine.h:class RanshiEngine: public HepRandomEngine {


review curand
----------------


* https://arxiv.org/pdf/1307.5869.pdf
* http://docs.nvidia.com/cuda/curand/host-api-overview.html#host-api-overview



feeding sequences to NonRandomEngine
---------------------------------------

::

   g4-;g4-cls NonRandomEngine


cfg4/tests/G4UniformRandTest.cc::

     08 int main(int argc, char** argv)
      9 {   
     10     PLOG_(argc, argv);
     11     
     12     LOG(info) << argv[0] ;
     13 
     14     
     15     unsigned N = 10 ;    // need to provide all that are consumed
     16     std::vector<double> seq ; 
     17     for(unsigned i=0 ; i < N ; i++ ) seq.push_back( double(i)/double(N) );
     18     
     19         
     20     long custom_seed = 9876 ;
     21     //CLHEP::HepJamesRandom* custom_engine = new CLHEP::HepJamesRandom();
     22     //CLHEP::MTwistEngine*   custom_engine = new CLHEP::MTwistEngine();
     23     
     24     CLHEP::NonRandomEngine*   custom_engine = new CLHEP::NonRandomEngine();
     25     custom_engine->setRandomSequence( seq.data(), seq.size() ) ; 
     26     
     27     CLHEP::HepRandom::setTheEngine( custom_engine );
     28     CLHEP::HepRandom::setTheSeed( custom_seed );    // does nothing for NonRandom
     29     
     30     CLHEP::HepRandomEngine* engine = CLHEP::HepRandom::getTheEngine() ;
     31     
     32     
     33     long seed = engine->getSeed() ;
     34     LOG(info) << " seed " << seed 
     35               << " name " << engine->name()
     36             ; 
     37             
     38     for(int i=0 ; i < 10 ; i++)
     39     {
     40         double u = engine->flat() ;   // equivalent to the standardly used: G4UniformRand() 
     41         std::cout << u << std::endl ;
     42     }   
     43     return 0 ;
     44 }   



curand same on host and device
--------------------------------

* https://devtalk.nvidia.com/default/topic/498171/how-to-get-same-output-by-curand-in-cpu-and-gpu/


::

    The quick answer: the simplest way to get the same results on the CPU and GPU
    is to use the host API. This allows you to generate random values into memory
    on the CPU or the GPU, the only difference is whether you call
    curandCreateGeneratorHost() versus curandCreateGenerator().

    To get the same results from the host API and the device API is a bit more
    work, you have to set things up carefully. The basic idea is that
    mathematically there is one long sequence of pseudorandom numbers. This long
    sequence is then cut up into chunks and shuffled together to get a final
    sequence that can be generated in parallel.


trying to get host and device curand to give same results
-----------------------------------------------------------


* matches in slice 0:4096 
* beyond that there is wrap back to the 2nd of 0

* http://docs.nvidia.com/cuda/curand/host-api-overview.html


::

    simon:cudarap blyth$ thrap-print 4095
    thrust_curand_printf
     i0 4095 i1 4096
     id:4095 thread_offset:0 
     0.841588  0.323815  0.475285  0.095566 
     0.397367  0.278207  0.916550  0.810093 
     0.764197  0.476796  0.743895  0.247211 
     0.946511  0.606670  0.736264  0.540743 
    curand_aligned_host
    generate NJ 16 clumps of NI 100000 :  0  1  2  3  4  5  6  7  8  9  10  11  12  13  14  15 
    dump i0:4095 i1:4096 
    i:4095 
    0.841588 0.323815 0.475285 0.095566 
    0.397367 0.278207 0.916550 0.810093 
    0.764197 0.476796 0.743895 0.247211 
    0.946511 0.606670 0.736264 0.540743 
    simon:cudarap blyth$ 
    simon:cudarap blyth$ 
    simon:cudarap blyth$ thrap-print 4096
    thrust_curand_printf
     i0 4096 i1 4097
     id:4096 thread_offset:0 
     0.840685  0.721466  0.500177  0.611869 
     0.970565  0.784008  0.867048  0.428319 
     0.040957  0.309976  0.847280  0.993939 
     0.238374  0.209762  0.010906  0.323518 
    curand_aligned_host
    generate NJ 16 clumps of NI 100000 :  0  1  2  3  4  5  6  7  8  9  10  11  12  13  14  15 
    dump i0:4096 i1:4097 
    i:4096 
    *0.438451* 0.517013 0.156989 0.071368 
    0.462508 0.227643 0.329358 0.144065 
    0.187799 0.915383 0.540125 0.974661 
    0.547469 0.653160 0.230238 0.338856 
    simon:cudarap blyth$ 


    ## beyond 4096 ... getting wrap back 

    simon:cudarap blyth$ thrap-print 0
    thrust_curand_printf
     i0 0 i1 1
     id:   0 thread_offset:0 
     0.740219 *0.438451*  0.517013  0.156989 
     0.071368  0.462508  0.227643  0.329358 
     0.144065  0.187799  0.915383  0.540125 
     0.974661  0.547469  0.653160  0.230238 
    curand_aligned_host
    generate NJ 16 clumps of NI 100000 :  0  1  2  3  4  5  6  7  8  9  10  11  12  13  14  15 
    dump i0:0 i1:1 
    i:0 
    0.740219 0.438451 0.517013 0.156989 
    0.071368 0.462508 0.227643 0.329358 
    0.144065 0.187799 0.915383 0.540125 
    0.974661 0.547469 0.653160 0.230238 
    simon:cudarap blyth$ 







reproduce zrntest subsequences with standanlone thrust_curand_printf
-----------------------------------------------------------------------

Using the known curand_init parameters for each photon_id used by cudarap- machinery 
that prepares the persisted rng_states are able to reproduce zrngtest
subsequences.

thrusttap/tests/thrust_curand_printf.cu::

     05 #include <thrust/for_each.h>
      6 #include <thrust/iterator/counting_iterator.h>
      7 #include <curand_kernel.h> 
      8 #include <iostream> 
      9 #include <iomanip>  
     10 
     11 struct curand_printf
     12 { 
     13     unsigned long long _seed ;
     14     unsigned long long _offset ;
     15     
     16     curand_printf( unsigned long long seed , unsigned long long offset )
     17        :
     18        _seed(seed),
     19        _offset(offset)
     20     {  
     21     }  
     22     
     23     __device__
     24     void operator()(unsigned id)
     25     { 
     26         unsigned int N = 16; // samples per thread 
     27         unsigned thread_offset = 0 ;
     28         curandState s;
     29         curand_init(_seed, id + thread_offset, _offset, &s);
     30         printf(" id:%4u thread_offset:%u \n", id, thread_offset );
     31         for(unsigned i = 0; i < N; ++i) 
     32         { 
     33             float x = curand_uniform(&s);
     34             printf(" %10.4f ", x );  
     35             if( i % 4 == 3 ) printf("\n") ;
     36         }   
     37     }   
     38 };  
     39 
     40 int main(int argc, char** argv)
     41 { 
     42      int id0 = argc > 1 ? atoi(argv[1]) : 0 ;
     43      int id1 = argc > 2 ? atoi(argv[2]) : 1 ;
     44      std::cout
     45          << " id0 " << id0
     46          << " id1 " << id1
     47          << std::endl  
     48          ;  
     49      thrust::for_each(
     50                 thrust::counting_iterator<int>(id0),
     51                 thrust::counting_iterator<int>(id1),
     52                 curand_printf(0,0));
     53     cudaDeviceSynchronize();
     54     return 0;
     55 }   





::

    simon:tests blyth$ thrust_curand_printf 0 1
     id0 0 id1 1
     id:   0 thread_offset:0 
         0.7402      0.4385      0.5170      0.1570 
         0.0714      0.4625      0.2276      0.3294 
         0.1441      0.1878      0.9154      0.5401 
         0.9747      0.5475      0.6532      0.2302 

    simon:tests blyth$ thrust_curand_printf 1 2
     id0 1 id1 2
     id:   1 thread_offset:0 
         0.9210      0.4604      0.3335      0.3725 
         0.4896      0.5673      0.0799      0.2334 
         0.5094      0.0890      0.0067      0.9542 
         0.5467      0.8245      0.5271      0.9301 

    simon:tests blyth$ thrust_curand_printf 2 3
     id0 2 id1 3
     id:   2 thread_offset:0 
         0.0390      0.2502      0.1845      0.9624 
         0.5206      0.9400      0.8306      0.4097 
         0.0816      0.8068      0.6953      0.6177 
         0.2563      0.2137      0.3424      0.2241 


    simon:tests blyth$ thrust_curand_printf 99997 99998 
     id0 99997 id1 99998
     id:99997 thread_offset:0 
         0.8181      0.2017      0.5459      0.0413 
         0.3800      0.9185      0.0232      0.0525 
         0.1143      0.7752      0.4034      0.9754 
         0.4632      0.8001      0.6522      0.7319 

    simon:tests blyth$ thrust_curand_printf 99998 99999 
     id0 99998 id1 99999
     id:99998 thread_offset:0 
         0.6275      0.0532      0.3444      0.8562 
         0.2001      0.3858      0.3199      0.4060 
         0.4550      0.9791      0.6474      0.8150 
         0.8287      0.0093      0.8172      0.5769 

    simon:tests blyth$ thrust_curand_printf 99999 100000
     id0 99999 id1 100000
     id:99999 thread_offset:0 
         0.9140      0.4403      0.9478      0.0900 
         0.9587      0.9880      0.2275      0.0438 
         0.7774      0.5031      0.3051      0.1865 
         0.3226      0.7396      0.6332      0.6526 





    simon:cudarap blyth$ curand_aligned_host 99997 100000
    j:0 generate NI 100000 
    j:1 generate NI 100000 
    j:2 generate NI 100000 
    j:3 generate NI 100000 
    j:4 generate NI 100000 
    j:5 generate NI 100000 
    j:6 generate NI 100000 
    j:7 generate NI 100000 
    j:8 generate NI 100000 
    j:9 generate NI 100000 
    j:10 generate NI 100000 
    j:11 generate NI 100000 
    j:12 generate NI 100000 
    j:13 generate NI 100000 
    j:14 generate NI 100000 
    j:15 generate NI 100000 
    dump i0:99997 i1:100000 
    i:99997 
    0.147038 0.798850 0.013086 0.858024 
    0.647867 0.735839 0.187833 0.655069 
    0.282454 0.655068 0.556091 0.426581 
    0.167576 0.321348 0.079367 0.099285 
    i:99998 
    0.786790 0.184093 0.507811 0.736662 
    0.317718 0.859347 0.905009 0.908526 
    0.860293 0.958224 0.112510 0.483687 
    0.052960 0.573791 0.291022 0.822895 
    i:99999 
    0.483006 0.974604 0.297720 0.621909 
    0.537028 0.619278 0.449021 0.444462 
    0.742229 0.548157 0.034401 0.118713 
    0.313563 0.877223 0.592213 0.742550 
    simon:cudarap blyth$ 




zrngtest : save 16 curand_uniform into photon buffer
--------------------------------------------------

* need to get the below zrngtest subsequences of randoms CPU side, 
  so can feed to NonRandomEngine ?

* hmm perhaps just grab from GPU ? but problem is do not know the 
  maximum number of rands needed for each photon 
  (actually it will depend on the bouncemax truncation configured)


oxrap/cu/generate.cu::

    264 RT_PROGRAM void zrngtest()
    265 {
    266     unsigned long long photon_id = launch_index.x ;
    267     unsigned int photon_offset = photon_id*PNUMQUAD ;
    268 
    269     curandState rng = rng_states[photon_id];
    270 
    271     photon_buffer[photon_offset+0] = make_float4(  curand_uniform(&rng) , curand_uniform(&rng) , curand_uniform(&rng), curand_uniform(&rng) );
    272     photon_buffer[photon_offset+1] = make_float4(  curand_uniform(&rng) , curand_uniform(&rng) , curand_uniform(&rng), curand_uniform(&rng) );
    273     photon_buffer[photon_offset+2] = make_float4(  curand_uniform(&rng) , curand_uniform(&rng) , curand_uniform(&rng), curand_uniform(&rng) );
    274     photon_buffer[photon_offset+3] = make_float4(  curand_uniform(&rng) , curand_uniform(&rng) , curand_uniform(&rng), curand_uniform(&rng) );
    275 
    276     rng_states[photon_id] = rng ;  // suspect this does nothing in my usage
    277 }


This is using saved rng_states cudarap/cuRANDWrapper_kernel.cu::

    093 __global__ void init_rng(int threads_per_launch, int thread_offset, curandState* rng_states, unsigned long long seed, unsigned long long offset)
    094 {
    ...
    110    int id = blockIdx.x*blockDim.x + threadIdx.x;
    111    if (id >= threads_per_launch) return;
    112 
    113    curand_init(seed, id + thread_offset , offset, &rng_states[id]);
    114 
    115    // not &rng_states[id+thread_offset] as rng_states is offset already in kernel call
    ...
    122 }


seed and offset both zero, from the filenames::

    simon:cfg4 blyth$ l /usr/local/opticks/installcache/RNG/
    total 258696
    -rw-r--r--  1 blyth  staff     450560 Jun 14 16:23 cuRANDWrapper_10240_0_0.bin
    -rw-r--r--  1 blyth  staff  132000000 Jun 14 16:23 cuRANDWrapper_3000000_0_0.bin
    simon:cfg4 blyth$ 


::

    tboolean-;tboolean-box --zrngtest 

    simon:tests blyth$ ls -l /tmp/blyth/opticks/evt/tboolean-box/torch/1/ox.npy
    -rw-r--r--  1 blyth  wheel  6400080 Dec  2 14:28 /tmp/blyth/opticks/evt/tboolean-box/torch/1/ox.npy

    tboolean-;TBOOLEAN_TAG=2 tboolean-box --zrngtest 

    simon:cfg4 blyth$ ls -l /tmp/blyth/opticks/evt/tboolean-box/torch/2/ox.npy
    -rw-r--r--  1 blyth  wheel  6400080 Dec  2 14:35 /tmp/blyth/opticks/evt/tboolean-box/torch/2/ox.npy





    simon:cudarap blyth$ curand_aligned_host 0 3
    j:0 generate NI 100000 
    j:1 generate NI 100000 
    j:2 generate NI 100000 
    j:3 generate NI 100000 
    j:4 generate NI 100000 
    j:5 generate NI 100000 
    j:6 generate NI 100000 
    j:7 generate NI 100000 
    j:8 generate NI 100000 
    j:9 generate NI 100000 
    j:10 generate NI 100000 
    j:11 generate NI 100000 
    j:12 generate NI 100000 
    j:13 generate NI 100000 
    j:14 generate NI 100000 
    j:15 generate NI 100000 
    dump i0:0 i1:3 
    i:0 
    0.740219 0.438451 0.517013 0.156989 
    0.071368 0.462508 0.227643 0.329358 
    0.144065 0.187799 0.915383 0.540125 
    0.974661 0.547469 0.653160 0.230238 
    i:1 
    0.920994 0.460364 0.333464 0.372520 
    0.489602 0.567271 0.079906 0.233368 
    0.509378 0.088979 0.006710 0.954227 
    0.546711 0.824547 0.527063 0.930132 
    i:2 
    0.039020 0.250215 0.184484 0.962422 
    0.520555 0.939965 0.830578 0.409733 
    0.081622 0.806771 0.695286 0.617707 
    0.256335 0.213682 0.342424 0.224079 
    simon:cudarap blyth$ 



This shows the reproducibility of the sequences::

    In [1]: import numpy as np

    In [2]: a = np.load("/tmp/blyth/opticks/evt/tboolean-box/torch/1/ox.npy")

    In [3]: a
    Out[3]: 
    array([[[ 0.74021935,  0.43845114,  0.51701266,  0.15698862],
            [ 0.07136751,  0.46250838,  0.22764327,  0.32935849],
            [ 0.14406531,  0.18779911,  0.91538346,  0.54012483],
            [ 0.97466087,  0.54746926,  0.65316027,  0.23023781]],

           [[ 0.9209938 ,  0.46036443,  0.33346406,  0.37252042],
            [ 0.48960248,  0.56727093,  0.07990581,  0.23336816],
            [ 0.50937784,  0.08897854,  0.00670976,  0.95422709],
            [ 0.54671133,  0.82454693,  0.52706289,  0.93013161]],

           [[ 0.03902049,  0.25021473,  0.18448432,  0.96242225],
            [ 0.5205546 ,  0.93996495,  0.83057821,  0.40973285],
            [ 0.08162197,  0.80677092,  0.69528568,  0.61770737],
            [ 0.25633496,  0.21368156,  0.34242383,  0.22407883]],

           ..., 
           [[ 0.81814659,  0.20170131,  0.54593664,  0.04129851],
            [ 0.38002208,  0.91853744,  0.02320537,  0.05250723],
            [ 0.11425403,  0.77515221,  0.40338024,  0.97540855],
            [ 0.46321765,  0.80014837,  0.65215546,  0.73192346]],

           [[ 0.62748933,  0.05319326,  0.34443355,  0.8561672 ],
            [ 0.2001164 ,  0.3857657 ,  0.31989732,  0.40597615],
            [ 0.45497316,  0.97913557,  0.64739084,  0.81499505],
            [ 0.82874513,  0.009322  ,  0.81717068,  0.57686758]],

           [[ 0.91401154,  0.44032493,  0.94783556,  0.09001808],
            [ 0.9587481 ,  0.98795038,  0.2274524 ,  0.04384946],
            [ 0.77744925,  0.50308371,  0.30509573,  0.18650141],
            [ 0.32255048,  0.73956126,  0.63323611,  0.65263885]]], dtype=float32)

    In [4]: a.min()
    Out[4]: 5.6193676e-07

    In [5]: a.max()
    Out[5]: 0.99999988

    In [6]: a.shape
    Out[6]: (100000, 4, 4)

    In [7]: b = np.load("/tmp/blyth/opticks/evt/tboolean-box/torch/2/ox.npy")

    In [8]: np.all( a == b )
    Out[8]: True





