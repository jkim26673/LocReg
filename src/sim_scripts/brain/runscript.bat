@echo off
:: Set up a loop that will run until the script completes successfully
:loop
echo Starting script at %date% %time%
call "C:\Users\kimjosy\AppData\Local\anaconda3\Scripts\activate.bat" C:\Users\kimjosy\AppData\Local\anaconda3
call conda activate myenv

:: Run your Python script
@REM python "C:\Users\kimjosy\Downloads\LocReg_Regularization-1\Simulations\brainscripts\cleanbraindatascript.py"
python "C:\Users\kimjosy\Downloads\LocReg_Regularization-1\Simulations\brainscripts\polishedbraindatascript.py"
:: Check if the script completed successfully
if %errorlevel% neq 0 (
    echo Script failed with error code %errorlevel%. Restarting...
    timeout /t 10 /nobreak
    goto loop
)

:: If script finished successfully, deactivate the environment
call conda deactivate

:: Log successful completion and exit
echo Script completed successfully at %date% %time%.
exit
