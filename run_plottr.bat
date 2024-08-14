@echo off
REM Activate the Conda environment
call conda activate py3.11

REM Run plottr-inspectr with the provided database path
plottr-inspectr --dbpath=%1