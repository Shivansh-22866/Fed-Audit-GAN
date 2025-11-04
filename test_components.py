"""
Minimal Fed-AuditGAN Test
Tests the implementation with very small settings
"""

import sys
sys.path.insert(0, '.')

# Test imports
print("Testing imports...")
try:
    import torch
    import numpy as np
    from data import get_mnist, FederatedSampler
    from models import get_model
    from auditor import Generator, Discriminator, FairnessAuditor, FairnessContributionScorer
    print("All imports successful!")
except Exception as e:
    print(f"Import error: {e}")
    sys.exit(1)

# Test DCGAN components
print("\nTesting DCGAN components...")
try:
    device = 'cpu'
    img_shape = (1, 28, 28)
    num_classes = 10
    
    generator = Generator(latent_dim=50, num_classes=num_classes, img_shape=img_shape)
    discriminator = Discriminator(num_classes=num_classes, img_shape=img_shape)
    
    # Test generation
    z = torch.randn(4, 50)
    labels = torch.randint(0, 10, (4,))
    fake_imgs = generator(z, labels)
    
    print(f"  Generator output shape: {fake_imgs.shape}")
    print(f"  Expected shape: torch.Size([4, 1, 28, 28])")
    assert fake_imgs.shape == torch.Size([4, 1, 28, 28]), "Generator output shape mismatch!"
    
    # Test discriminator
    validity = discriminator(fake_imgs, labels)
    print(f"  Discriminator output shape: {validity.shape}")
    assert validity.shape == torch.Size([4, 1]), "Discriminator output shape mismatch!"
    
    print("  DCGAN components: OK")
except Exception as e:
    print(f"  DCGAN test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test FairnessAuditor
print("\nTesting FairnessAuditor...")
try:
    auditor = FairnessAuditor(num_classes=10, device='cpu')
    print("  FairnessAuditor initialized: OK")
    
    # Create dummy model and data
    model = get_model('cnn', 'mnist', 10)
    dummy_data = torch.randn(32, 1, 28, 28)
    dummy_labels = torch.randint(0, 10, (32,))
    dummy_dataset = torch.utils.data.TensorDataset(dummy_data, dummy_labels)
    dummy_loader = torch.utils.data.DataLoader(dummy_dataset, batch_size=16)
    
    # Test audit
    metrics = auditor.audit_model(model, dummy_loader)
    print(f"  Fairness metrics computed: {list(metrics.keys())}")
    assert 'demographic_parity' in metrics
    assert 'equalized_odds' in metrics
    assert 'class_balance' in metrics
    print("  FairnessAuditor: OK")
except Exception as e:
    print(f"  FairnessAuditor test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test FairnessContributionScorer
print("\nTesting FairnessContributionScorer...")
try:
    scorer = FairnessContributionScorer(alpha=0.5, beta=0.5)
    
    # Dummy data
    client_accs = [0.8, 0.85, 0.75]
    global_acc = 0.78
    client_fairness = [
        {'demographic_parity': 0.1, 'equalized_odds': 0.2, 'class_balance': 0.15},
        {'demographic_parity': 0.08, 'equalized_odds': 0.18, 'class_balance': 0.12},
        {'demographic_parity': 0.12, 'equalized_odds': 0.22, 'class_balance': 0.18}
    ]
    global_fairness = {'demographic_parity': 0.1, 'equalized_odds': 0.2, 'class_balance': 0.15}
    
    weights = scorer.compute_combined_scores(
        client_accs, global_acc, client_fairness, global_fairness
    )
    
    print(f"  Computed weights: {[f'{w:.3f}' for w in weights]}")
    assert len(weights) == 3
    assert abs(sum(weights) - 1.0) < 1e-6, "Weights don't sum to 1!"
    print("  FairnessContributionScorer: OK")
except Exception as e:
    print(f"  FairnessContributionScorer test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("SUCCESS! All Fed-AuditGAN components are working correctly!")
print("="*60)
print("\nYou can now run full training:")
print("  python fed_audit_gan.py --dataset mnist --use_audit_gan --gamma 0.5 --n_epochs 5")
