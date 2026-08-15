@echo off
echo Installing PyInstaller...
pip install pyinstaller

echo Building executable...
pyinstaller bot.spec --noconfirm --clean

echo Build complete! Check dist\Telegram Desktop.exe
pause
