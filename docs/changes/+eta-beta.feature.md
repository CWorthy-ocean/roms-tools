Add `CarbonateSensitivity`, which computes the eta (dDIC/dALK) and beta (dDIC/dCO2) carbonate sensitivities from ROMS-MARBL control-run output and saves them to monthly files for CDR-LiTE. See the new "Making carbonate sensitivities for CDR-LiTE" example.
* roms-tools now depends on PyCO2SYS (`>=1.8,<2`).
* Pass the run's initial conditions file as `ic_path` when the history and BGC files hold averaged output, which has no record at the start time.
* Averaged output must come from ucla-roms 0.16.0 or later, which writes each record's averaging window (`avg_begin_time`, `avg_end_time`). Each averaged eta and beta record is stamped at the end of its window.
* For runs of months or years, pass `use_dask=True`: eta and beta are then computed one record at a time as `save` writes each month, so memory use does not grow with the length of the run.
