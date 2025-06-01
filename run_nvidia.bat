@echo off
echo Starting AI Image Processor (CUDA)...
call venv\Scripts\activate.bat
set CUDA_VISIBLE_DEVICES=0
python main.py
pause
