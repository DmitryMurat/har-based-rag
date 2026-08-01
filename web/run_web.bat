@echo off
REM Двойной клик в Проводнике запускает веб-интерфейс RAG в фоне без окна консоли
REM (используем pythonw.exe — у него нет консоли в принципе) и открывает страницу
REM в браузере. Окно cmd мелькает на долю секунды и сразу закрывается — сам сервер
REM живёт отдельно от него. Вывод и ошибки пишутся в web\server.log.
cd /d "%~dp0\.."
if exist venv\Scripts\pythonw.exe (
    start "" venv\Scripts\pythonw.exe -u web\web.py > web\server.log 2>&1
) else (
    start "" venv\Scripts\python.exe -u web\web.py > web\server.log 2>&1
)
exit
