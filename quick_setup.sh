#!/bin/bash
# Quick Setup Script for Fed-AuditGAN
# Installs dependencies using pip

echo "======================================================================"
echo "Fed-AuditGAN Quick Setup"
echo "======================================================================"
echo ""
echo "Installing PyTorch and dependencies..."
echo ""

# Install PyTorch CPU version (faster to install)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install numpy matplotlib tqdm

echo ""
echo "======================================================================"
echo "Verifying installation..."
echo "======================================================================"
echo ""

# Verify installation
python -c "import torch; print('✅ PyTorch version:', torch.__version__)"
python -c "import numpy; print('✅ NumPy version:', numpy.__version__)"
python -c "import torchvision; print('✅ TorchVision version:', torchvision.__version__)"
python -c "import matplotlib; print('✅ Matplotlib installed')"
python -c "import tqdm; print('✅ tqdm installed')"

echo ""
echo "======================================================================"
echo "✅ Setup Complete!"
echo "======================================================================"
echo ""
echo "You can now run Fed-AuditGAN:"
echo "  python fed_audit_gan.py --dataset mnist --use_audit_gan --gamma 0.5"
echo ""
