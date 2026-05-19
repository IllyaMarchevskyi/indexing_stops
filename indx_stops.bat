set "PROJECT_DIR=C:\Users\User\Documents\Program\indx_stops"  
set "PYTHON_EXE=.\.venv\Scripts\python.exe"  
set "indx_str=300"  

cd /d "%PROJECT_DIR%" || (  
    echo not found folder:  
    echo %PROJECT_DIR%  
    pause  
    exit /b 1  
)  

"%PYTHON_EXE%" match_stops.py --auto-select-exact --start-osm-row-id %indx_str% --open-browser --pyppeteer-executable-path "C:\Program Files\Google\Chrome\Application\chrome.exe"  


if errorlevel 1 (  
    echo.  
    echo Скрипт завершився з помилкою.  
    pause  
)  

endlocal