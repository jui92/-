@echo off
chcp 65001 >nul

echo ===============================================
echo       GraphRAG Indexing Automation Script
echo ===============================================

REM 항상 디버깅 모드로 시작
echo [DEBUG] Script started. Current directory: %cd%
echo Press any key to continue...
pause >nul

REM 1) Conda 환경 체크
echo [1/5] Checking Conda environment...
CALL conda info --envs | findstr "audit_graphrag" >nul
IF ERRORLEVEL 1 (
    echo.
    echo [ERROR] Conda environment 'audit_graphrag' not found.
    echo The window will not close automatically.
    pause
    goto END
)
echo [OK] Found conda env: audit_graphrag

REM 2) Conda 환경 활성화
echo [2/5] Activating Conda...
CALL conda activate audit_graphrag
IF ERRORLEVEL 1 (
    echo.
    echo [ERROR] Failed to activate conda environment.
    pause
    goto END
)

REM 3) .env 로딩
echo [3/5] Loading .env file...
if not exist ".env" (
    echo [ERROR] .env file not found!
    pause
    goto END
)

for /f "usebackq tokens=* delims=" %%a in (".env") do (
    set %%a
)

echo Loaded OPENAI_API_KEY: %OPENAI_API_KEY%

REM 4) GraphRAG 실행
echo.
echo [4/5] Running GraphRAG Indexing...
echo Start time: %date% %time%
echo -----------------------------------------------

python -m graphrag.cli.index --root "%cd%" --verbose
set EXITCODE=%ERRORLEVEL%

echo -----------------------------------------------
echo Python exit code: %EXITCODE%

IF %EXITCODE% NEQ 0 (
    echo.
    echo [ERROR] GraphRAG failed.
    echo Check the error above.
    pause
    goto END
)

echo.
echo [5/5] Finished successfully!
pause

:END
echo Script ended.
pause
