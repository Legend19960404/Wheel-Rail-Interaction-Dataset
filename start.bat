@echo off
REM 设置环境变量
echo setting enviroment variables ...
set PYTHON_HOME=..
set PATH=%PYTHON_HOME%;%PYTHON_HOME%\Library\mingw-w64\bin;%PYTHON_HOME%\Library\usr\bin;%PYTHON_HOME%\Library\bin;%PYTHON_HOME%\Scripts;%PYTHON_HOME%\bin;%PATH%
set CONTACT_LIB=%PYTHON_HOME%\Lib\site-packages\bin
REM 检查 dataset_config.json 文件是否存在
if not exist "%1" (
    echo Error: configuration file not found in current directory.
    echo Current directory: %CD%
    pause
)
echo executing database_steady_rolling.py ...
python database_steady_rolling.py %1