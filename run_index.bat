@echo off
chcp 65001 >nul

echo ===============================================
echo       GraphRAG Indexing Automation Script
echo ===============================================

echo [DEBUG] Script started. Current directory: %cd%
pause

REM 1) Conda 환경 체크
echo [1/5] Checking Conda environment...
CALL conda info --envs | findstr "audit_graphrag" >nul
IF ERRORLEVEL 1 (
    echo [ERROR] Conda environment 'audit_graphrag' not found.
    pause
    exit /b 1
)
echo [OK] Found conda env: audit_graphrag

REM 2) Conda 환경 활성화
echo [2/5] Activating Conda...
CALL conda activate audit_graphrag

REM 3) .env 로딩
echo [3/5] Loading .env file...
for /f "usebackq tokens=* delims=" %%a in (".env") do (
    set %%a
)
echo Loaded OPENAI_API_KEY: %OPENAI_API_KEY%

REM 4) GraphRAG 실행
echo.
echo [4/5] Running GraphRAG Indexing...
echo Start time: %date% %time%

CALL graphrag index --root "%cd%" --verbose

IF ERRORLEVEL 1 (
    echo.
    echo [ERROR] GraphRAG failed.
    pause
    exit /b 1
)

echo.
echo [5/5] Finished successfully!
pause
