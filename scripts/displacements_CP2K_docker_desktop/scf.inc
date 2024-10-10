      SCF_GUESS RESTART
      CHOLESKY  RESTORE # INVERSE
      EPS_SCF 5E-9
      MAX_SCF 30 #50 or without outer_scf 500
      IGNORE_CONVERGENCE_FAILURE T
     
      &OT T # this or DIAGONALIZATION
        MINIMIZER DIIS # CG
        PRECONDITIONER FULL_ALL #FULL_KINETIC
      &END OT
      
      &DIAGONALIZATION  OFF # this or OT
        ALGORITHM STANDARD
      &END DIAGONALIZATION

      &MIXING  T
        METHOD BROYDEN_MIXING
        ALPHA 0.4
        NBROYDEN 8
      &END MIXING


      &OUTER_SCF  T
         OPTIMIZER  SD #DIIS
         EPS_SCF     5E-9
         MAX_SCF  50
      &END OUTER_SCF

      &PRINT
          &RESTART  SILENT
           ADD_LAST  NUMERIC
           &EACH
             QS_SCF  20
           &END EACH
         &END RESTART
       &END PRINT
