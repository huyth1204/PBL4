@echo off
chcp 65001 >nul
cd /d "%~dp0.."
echo Cai thu vien (chi can lan dau)...
python -m pip install -r huong_dan\yeu_cau.txt
echo Chay 10 buoc, ket qua mo trong trinh duyet...
python huong_dan\chay_tat_ca.py
pause
