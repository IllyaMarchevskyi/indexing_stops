Install Python for windows

```
powershell -ExecutionPolicy Bypass -File .\install_python.ps1
```

Install python lib
```
pip install -r requirements.txt
```

Run code for windows
```
python match_stops.py --auto-select-exact --open-browser --pyppeteer-executable-path "C:\Program Files\Google\Chrome\Application\chrome.exe"
```