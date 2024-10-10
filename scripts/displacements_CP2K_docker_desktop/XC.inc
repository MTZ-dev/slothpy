       &XC_GRID
         XC_SMOOTH_RHO  NN50
         XC_DERIV  NN50_SMOOTH
       &END XC_GRID

  &XC_FUNCTIONAL PBE
  &END XC_FUNCTIONAL 
 
  &vDW_POTENTIAL 
     POTENTIAL_TYPE  PAIR_POTENTIAL
     &PAIR_POTENTIAL
       TYPE DFTD3
       PARAMETER_FILE_NAME dftd3.dat 
       REFERENCE_FUNCTIONAL  PBE
       VERBOSE_OUTPUT .TRUE.
     &END PAIR_POTENTIAL
  &END vDW_POTENTIAL

#  &HF # Set this section for periodic Hartree Fock calculation e.g. when using hybrid functional
#    &INTERACTION_POTENTIAL
#       POTENTIAL_TYPE TRUNCATED
#       CUTOFF_RADIUS 6.0 # Cutoff radius (Angstrom) for truncated 1/r Coulomb operator. Set it to < half of the smallest unit cell parameter
#    &END INTERACTION_POTENTIAL
#    &SCREENING
#      EPS_SCHWARZ 1.0E-7 # Lower this for more precise calculations at high computational cost
#      EPS_SCHWARZ_FORCES 1.0E-5 # This will be approximately the accuracy of the forces so lower this as well (default 100*EPS_SCHWARZ)
#      # SCREEN_ON_INITIAL_P T # Screening ERI based on initial density matrix. You need to provide wavefunction restart file
#    &END SCREENING
#  &END HF
