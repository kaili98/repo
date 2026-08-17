@echo off
echo Installing PyInstaller...
pip install pyinstaller

echo Building executable...
py -m PyInstaller bot.spec --noconfirm --clean --distpath dist

echo Build complete! Check dist\Telegram Desktop.exe
pause
