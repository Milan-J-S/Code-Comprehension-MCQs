cd src
REM batch file to run the scripts for the scratch logger
REM starts the browser and navigates to localhost:5000

ECHO Please wait while the server starts up

start chrome "http://localhost:5000" && python server.py REM change the path here if required
