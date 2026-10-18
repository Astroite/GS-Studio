@echo off
setlocal
if not defined LP_ROOT set "LP_ROOT=E:\AI\LocalPostshot"
set "LP_PY_ROOT=%LP_ROOT%\runtime\python-install\cpython-3.10.22-windows-x86_64-none"
set "PATH=%LP_PY_ROOT%\Scripts;%PATH%"
if not defined LP_VCVARS set "LP_VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvarsall.bat"
call "%LP_VCVARS%" x64 -vcvars_ver=14.44
if errorlevel 1 exit /b %errorlevel%
cmake -S "%LP_ROOT%\sources\spz" -B "%LP_ROOT%\build\spz" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS=/utf-8 -DSPZ_BUILD_PYTHON_BINDINGS=ON -DSPZ_BUILD_TOOLS=OFF -DBUILD_SHARED_LIBS=OFF -DFETCHCONTENT_SOURCE_DIR_ZLIB="%LP_ROOT%\sources\zlib-1.3.2" -DFETCHCONTENT_SOURCE_DIR_ZSTD="%LP_ROOT%\sources\zstd" -DPython_EXECUTABLE="%LP_PY_ROOT%\python.exe" -DCMAKE_INSTALL_PREFIX="%LP_ROOT%\runtime\spz-install"
if errorlevel 1 exit /b %errorlevel%
cmake --build "%LP_ROOT%\build\spz" --parallel 6
if errorlevel 1 exit /b %errorlevel%
cmake --install "%LP_ROOT%\build\spz"
if errorlevel 1 exit /b %errorlevel%
if not exist "%LP_PY_ROOT%\Lib\site-packages\spz" mkdir "%LP_PY_ROOT%\Lib\site-packages\spz"
copy /Y "%LP_ROOT%\sources\spz\src\python\spz\__init__.py" "%LP_PY_ROOT%\Lib\site-packages\spz\__init__.py" >nul
for /R "%LP_ROOT%\runtime\spz-install" %%f in (*.pyd) do copy /Y "%%f" "%LP_PY_ROOT%\Lib\site-packages\spz\" >nul
for /R "%LP_ROOT%\runtime\spz-install" %%f in (*.dll) do copy /Y "%%f" "%LP_PY_ROOT%\Lib\site-packages\spz\" >nul
exit /b 0
