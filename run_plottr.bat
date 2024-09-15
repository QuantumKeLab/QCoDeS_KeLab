@echo off
REM Activate the Conda environment
call conda activate qcodes

REM Run plottr-inspectr with the provided database path
plottr-inspectr --dbpath=%1