@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul

echo ===============================================
echo   PDF → TXT 변환 + GraphRAG Indexing 자동화
echo ===============================================
echo 현재 작업 디렉토리: %cd%
echo.

REM ----------------------------------------------------
REM 1. Conda 환경 확인
REM ----------------------------------------------------
echo [1/6] Conda 환경 확인 중...
conda env list | findstr /c:"audit_graphrag" >nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] audit_graphrag 환경이 존재하지 않습니다.
    pause
    exit /b
)
echo [OK] audit_graphrag 환경 발견
echo.

REM ----------------------------------------------------
REM 2. Conda 환경 활성화 (정상 작동 방식)
REM ----------------------------------------------------
echo [2/6] Conda 환경 활성화...
call C:\Users\Administrator\anaconda3\Scripts\activate.bat audit_graphrag

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] conda activate 실패
    pause
    exit /b
)
echo [OK] conda activate 완료
echo.

REM ----------------------------------------------------
REM 3. .env 파일 로딩
REM ----------------------------------------------------
echo [3/6] .env 파일 로딩 중...
for /f "usebackq tokens=1,* delims==" %%a in (".env") do (
    set %%a=%%b
)
echo [OK] API Key 로딩 완료
echo.

REM ----------------------------------------------------
REM 4. PDF → TXT 변환 수행 (신규/없는 TXT만 처리)
REM ----------------------------------------------------
echo [4/6] PDF → TXT 변환 준비 중...
echo TXT는 없는 파일만 생성됩니다.
echo.

python scripts\pdf_to_txt.py input input_txt input

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] PDF → TXT 변환 실패
    pause
    exit /b
)
echo [OK] PDF → TXT 변환 완료
echo.

REM ----------------------------------------------------
REM 5. GraphRAG Indexing 실행
REM ----------------------------------------------------
echo [5/6] GraphRAG Indexing 실행...
echo 시작 시간: %date% %time%
echo.

graphrag index --root . --verbose

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] GraphRAG Indexing 실패
    pause
    exit /b
)

echo [OK] GraphRAG Indexing 완료!
echo.
pause
