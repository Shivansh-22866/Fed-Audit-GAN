@echo off
REM ============================================================================
REM Fed-Audit-GAN - GitHub Upload Helper Script
REM ============================================================================

echo.
echo ================================================================================
echo Fed-Audit-GAN - GitHub Upload Helper
echo ================================================================================
echo.
echo This script will help you upload Fed-Audit-GAN to GitHub.
echo.
echo STEP 1: Create GitHub Repository
echo ================================================================================
echo.
echo 1. Go to: https://github.com/new
echo 2. Repository name: Fed-Audit-GAN
echo 3. Description: Fairness-Aware Federated Learning with Generative Auditing
echo 4. Visibility: Public
echo 5. DO NOT initialize with README, .gitignore, or license
echo 6. Click "Create repository"
echo.
pause
echo.

echo STEP 2: Enter Your GitHub Username
echo ================================================================================
echo.
set /p username="Enter your GitHub username (default: 99VICKY99): "
if "%username%"=="" set username=99VICKY99
echo.
echo Using username: %username%
echo.

echo STEP 3: Connecting to GitHub
echo ================================================================================
echo.

REM Check if remote exists
git remote get-url origin >nul 2>&1
if %errorlevel% equ 0 (
    echo Remote 'origin' already exists. Removing...
    git remote remove origin
)

REM Add remote
echo Adding GitHub repository as remote...
git remote add origin https://github.com/%username%/Fed-Audit-GAN.git
echo.

REM Rename branch to main
echo Renaming branch to 'main'...
git branch -M main
echo.

echo STEP 4: Pushing to GitHub
echo ================================================================================
echo.
echo Now pushing your code to GitHub...
echo You may be asked to authenticate.
echo.
echo If asked for password, use your Personal Access Token:
echo 1. Go to: https://github.com/settings/tokens
echo 2. Generate new token (classic)
echo 3. Select 'repo' scope
echo 4. Copy token and paste as password
echo.
pause
echo.

git push -u origin main

if %errorlevel% equ 0 (
    echo.
    echo ================================================================================
    echo SUCCESS! Your project is now on GitHub!
    echo ================================================================================
    echo.
    echo Repository URL: https://github.com/%username%/Fed-Audit-GAN
    echo.
    echo Opening repository in browser...
    start https://github.com/%username%/Fed-Audit-GAN
    echo.
) else (
    echo.
    echo ================================================================================
    echo Upload failed. Please check:
    echo ================================================================================
    echo.
    echo 1. Did you create the repository on GitHub?
    echo 2. Is your username correct? (%username%)
    echo 3. Did you authenticate correctly?
    echo.
    echo Read UPLOAD_TO_GITHUB.md for detailed instructions.
    echo.
)

pause
