Add `calculate_eta_beta`, which computes the eta (dDIC/dALK) and beta (dDIC/dCO2) carbonate sensitivities from ROMS-MARBL control-run output and saves them to monthly files for CDR-LiTE. See the new "Making carbonate sensitivities for CDR-LiTE" example.
* roms-tools now depends on PyCO2SYS (`>=1.8,<2`).
* Pass the run's initial conditions file as `ic_path` when the history and BGC files hold averaged output, which has no record at the start time.
