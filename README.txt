H. Österbom (2025) 

This project provides a complete CASA-based simulation pipeline for generating ALMA-like Measurement Sets (MSs), applying realistic noise, building per-antenna primary beams, performing imaging, and validating noise behavior.
It supports both homogeneous arrays (e.g. standard 12-m ALMA configurations) and heterogeneous arrays with arbitrary dish sizes.
ps: The pipeline has only been tested on CASA 6.7!

project structure:

run_simobserve_pipeline.py     # Main orchestration script
qol_simulation_pipeline.py     # Simulation + imaging core (MS creation, noise, SEFD, RMS tools)
generate_builtin_sky_model.py  # Built-in sky models (ring, point+disk)
pb_vp.py                       # PB / VP image and table generation
utils.py                       # Logging, cleanup, naming utilities
validation.py                  # Noise validation routines


For more information, read the README.pdf, or if you have any questions: hampusosterbom1@gmail.com