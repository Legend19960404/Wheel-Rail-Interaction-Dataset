@echo off
setlocal enabledelayedexpansion
REM =============== 前置检查 ===============
if not exist "start.bat" (
    echo [ERROR] start.bat not found in current directory.
    echo Current directory: %CD%
    pause
    exit /b 1
)

set COUNT=0
for %%f in (*.json) do set /a COUNT+=1

if %COUNT% EQU 0 (
    echo [ERROR] No JSON configuration files found.
    echo Current directory: %CD%
    pause
    exit /b 1
)
echo [INFO] Found %COUNT% JSON file(s). Starting tasks with max 5 concurrent processes...
REM =============== 任务调度 ===============
set MAX_JOBS=5
set INDEX=0
set BATCH_DIR=..
set "TEMP_DIR=%BATCH_DIR%\code\Task_Queue_%RANDOM%%RANDOM%"
mkdir "%TEMP_DIR%" 2>nul
goto :main_loop
:main_loop
for %%f in (*.json) do (
    set /a INDEX+=1
    set "LOCK=%TEMP_DIR%\task_!INDEX!.lock"
    
    REM 等待空闲槽位（仅检查锁文件数量）
    call :wait_slot
    
    REM 创建任务锁文件 + 启动任务（任务结束自动删除锁）
    type nul > "!LOCK!"
    start cmd /c "call start.bat "%%f" && del "!LOCK!" 2>nul || del "!LOCK!" 2>nul"
    
    echo [!INDEX!/!COUNT!] STARTED: %%f
)
goto :wait_completion
:wait_slot
    set ACTIVE=0
    for %%L in ("%TEMP_DIR%\*.lock") do set /a ACTIVE+=1
    if !ACTIVE! GEQ %MAX_JOBS% (
        timeout /t 1 /nobreak >nul
        goto :wait_slot
    )
    exit /b
:wait_completion
    timeout /t 2 /nobreak >nul
    set REMAIN=0
    for %%L in ("%TEMP_DIR%\*.lock") do set /a REMAIN+=1
    if !REMAIN! GTR 0 (
        goto :wait_completion
    )
    rmdir /s /q "%TEMP_DIR%" >nul 2>&1
	
    if not exist "%TEMP_DIR%" (
		echo [INFO] Temporary directory has been cleaned successfully!
    )
    echo [SUCCESS] All tasks completed!
	echo.
	pause
	exit /b 0



