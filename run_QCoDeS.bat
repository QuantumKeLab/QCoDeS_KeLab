@echo off
REM Activate the Conda environment
call conda activate qcodes

REM Run plottr-inspectr with the provided database path --dbpath=%1
start plottr-inspectr 

REM Start jupyter-notebook
jupyter notebook