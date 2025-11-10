@echo off
REM ============================================================================
REM Fed-AuditGAN Simple Launcher for Windows
REM ============================================================================
REM Direct Python execution without conda activation
REM ============================================================================

set PYTHON_PATH=C:\Users\vicky\anaconda3\envs\fedavg\python.exe

:MENU
cls
echo.
echo ================================================================================
echo Fed-AuditGAN: Fairness-Aware Federated Learning
echo ================================================================================
echo.
echo Select an experiment to run:
echo.
echo MNIST Quick Tests (2 rounds):
echo   [1] MNIST - Standard FedAvg - quick test
echo   [2] MNIST - Fed-AuditGAN gamma=0.5 - quick test
echo.
echo MNIST Gamma Comparison (50 rounds with WandB):
echo   [3] Run ALL gamma values - 0.0, 0.3, 0.5, 0.7, 1.0
echo   [4] Gamma=0.0 - Pure Accuracy - NO fairness optimization
echo   [5] Gamma=0.3 - Accuracy-Focused - 30% fairness, 70% accuracy
echo   [6] Gamma=0.5 - Balanced - 50% fairness, 50% accuracy
echo   [7] Gamma=0.7 - Fairness-Focused - 70% fairness, 30% accuracy
echo   [8] Gamma=1.0 - Pure Fairness - 100% fairness optimization
echo.
echo Standard Experiments:
echo   [9] MNIST - Standard FedAvg - No Fed-AuditGAN
echo   [C] CIFAR-10 - Fed-AuditGAN - gamma=0.5 balanced
echo.
echo Other:
echo   [Q] Quit
echo   [H] Help / Custom Parameters
echo.
set /p choice="Enter your choice: "

REM Process choice
if /i "%choice%"=="1" (
    echo Running MNIST - Standard FedAvg - quick test...
    "%PYTHON_PATH%" fed_audit_gan.py --dataset mnist --n_clients 3 --n_epochs 2 --n_client_epochs 1 --batch_size 32 --device cpu --exp_name "MNIST_FedAvg_test"
    goto END
)
if /i "%choice%"=="2" (
    echo Running MNIST - Fed-AuditGAN gamma=0.5 - quick test...
    "%PYTHON_PATH%" fed_audit_gan.py --dataset mnist --use_audit_gan --gamma 0.5 --n_clients 3 --n_epochs 2 --n_client_epochs 1 --n_audit_steps 3 --n_probes 100 --batch_size 32 --device cpu --exp_name "MNIST_AuditGAN_test"
    goto END
)
if /i "%choice%"=="3" (
    echo.
    echo ========================================================================
    echo Running COMPLETE GAMMA COMPARISON STUDY
    echo ========================================================================
    echo This will run 5 experiments sequentially:
    echo   1. Gamma=0.0 - Pure Accuracy
    echo   2. Gamma=0.3 - Accuracy-Focused  
    echo   3. Gamma=0.5 - Balanced
    echo   4. Gamma=0.7 - Fairness-Focused
    echo   5. Gamma=1.0 - Pure Fairness
    echo.
    echo Total time: ~5-10 hours
    echo Results will be logged to WandB for comparison
    echo.
    pause
    
    echo.
    echo [1/5] Running Gamma=0.0 - Pure Accuracy...
    "%PYTHON_PATH%" fed_audit_gan.py --dataset mnist --partition_mode iid --use_audit_gan --gamma 0.0 --n_epochs 50 --wandb --exp_name "MNIST_Gamma_0.0_PureAccuracy"
    
    echo.
    echo [2/5] Running Gamma=0.3 - Accuracy-Focused...
    "%PYTHON_PATH%" fed_audit_gan.py --dataset mnist --partition_mode iid --use_audit_gan --gamma 0.3 --n_epochs 50 --wandb --exp_name "MNIST_Gamma_0.3_AccuracyFocused"
    
    echo.
    echo [3/5] Running Gamma=0.5 - Balanced...
    "%PYTHON_PATH%" fed_audit_gan.py --dataset mnist --partition_mode iid --use_audit_gan --gamma 0.5 --n_epochs 50 --wandb --exp_name "MNIST_Gamma_0.5_Balanced"
    
    echo.
    echo [4/5] Running Gamma=0.7 - Fairness-Focused...
    "%PYTHON_PATH%" fed_audit_gan.py --dataset mnist --partition_mode iid --use_audit_gan --gamma 0.7 --n_epochs 50 --wandb --exp_name "MNIST_Gamma_0.7_FairnessFocused"
    
    echo.
    echo [5/5] Running Gamma=1.0 - Pure Fairness...
    "%PYTHON_PATH%" fed_audit_gan.py --dataset mnist --partition_mode iid --use_audit_gan --gamma 1.0 --n_epochs 50 --wandb --exp_name "MNIST_Gamma_1.0_PureFairness"
    
    echo.
    echo ========================================================================
    echo All 5 experiments complete!
    echo Compare results on WandB to see gamma impact
    echo ========================================================================
    goto END
)
if /i "%choice%"=="4" (
    echo Running Gamma=0.0 - Pure Accuracy - NO fairness optimization...
    echo This uses standard FedAvg with Fed-AuditGAN infrastructure but 100%% accuracy focus
    "%PYTHON_PATH%" fed_audit_gan.py --dataset mnist --partition_mode iid --use_audit_gan --gamma 0.0 --n_epochs 50 --wandb --exp_name "MNIST_Gamma_0.0"
    goto END
)
if /i "%choice%"=="5" (
    echo Running Gamma=0.3 - Accuracy-Focused - 30%% fairness, 70%% accuracy...
    "%PYTHON_PATH%" fed_audit_gan.py --dataset mnist --partition_mode iid --use_audit_gan --gamma 0.3 --n_epochs 50 --wandb --exp_name "MNIST_Gamma_0.3"
    goto END
)
if /i "%choice%"=="6" (
    echo Running Gamma=0.5 - Balanced - 50%% fairness, 50%% accuracy...
    "%PYTHON_PATH%" fed_audit_gan.py --dataset mnist --partition_mode iid --use_audit_gan --gamma 0.5 --n_epochs 50 --wandb --exp_name "MNIST_Gamma_0.5"
    goto END
)
if /i "%choice%"=="7" (
    echo Running Gamma=0.7 - Fairness-Focused - 70%% fairness, 30%% accuracy...
    "%PYTHON_PATH%" fed_audit_gan.py --dataset mnist --partition_mode iid --use_audit_gan --gamma 0.7 --n_epochs 50 --wandb --exp_name "MNIST_Gamma_0.7"
    goto END
)
if /i "%choice%"=="8" (
    echo Running Gamma=1.0 - Pure Fairness - 100%% fairness optimization...
    echo This maximizes fairness at the cost of some accuracy
    "%PYTHON_PATH%" fed_audit_gan.py --dataset mnist --partition_mode iid --use_audit_gan --gamma 1.0 --n_epochs 50 --wandb --exp_name "MNIST_Gamma_1.0"
    goto END
)
if /i "%choice%"=="9" (
    echo Running MNIST - Standard FedAvg - NO Fed-AuditGAN...
    "%PYTHON_PATH%" fed_audit_gan.py --dataset mnist --partition_mode iid --n_epochs 50 --exp_name "MNIST_FedAvg_Baseline"
    goto END
)
if /i "%choice%"=="C" (
    echo Running CIFAR-10 - Fed-AuditGAN - gamma=0.5 balanced...
    "%PYTHON_PATH%" fed_audit_gan.py --dataset cifar10 --partition_mode iid --use_audit_gan --gamma 0.5 --n_epochs 60 --wandb --exp_name "CIFAR10_Gamma_0.5"
    goto END
)
if /i "%choice%"=="Q" (
    echo Exiting...
    exit /b 0
)
if /i "%choice%"=="H" (
    echo.
    echo ========================================================================
    echo Custom Parameters Example:
    echo ========================================================================
    echo.
    echo "%PYTHON_PATH%" fed_audit_gan.py ^
    echo     --dataset mnist ^
    echo     --model_name cnn ^
    echo     --partition_mode shard ^
    echo     --n_clients 10 ^
    echo     --n_epochs 50 ^
    echo     --use_audit_gan ^
    echo     --gamma 0.5 ^
    echo     --n_audit_steps 100 ^
    echo     --exp_name "My_Experiment"
    echo.
    echo For full help: "%PYTHON_PATH%" fed_audit_gan.py --help
    echo.
    pause
    goto MENU
)

echo Invalid choice. Please try again.
pause
goto MENU

:END
echo.
echo ================================================================================
echo Experiment completed!
echo Results saved in ./results/
echo ================================================================================
echo.
pause
