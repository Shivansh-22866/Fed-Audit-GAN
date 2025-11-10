"""
Fed-AuditGAN: Fairness-Aware Federated Learning
===============================================

Implementation of the Fed-AuditGAN algorithm which combines federated learning
with generative fairness auditing to balance accuracy and fairness.

The algorithm consists of 4 phases per round:
1. Standard FL Training Round (clients train locally)
2. Generative Fairness Auditing (server trains generator to find biases)
3. Fairness Contribution Scoring (server scores each client)
4. Multi-Objective Aggregation (weighted aggregation based on fairness+accuracy)

Reference: Based on FairFed and Fed-AuditGAN concepts
"""

import torch
import torch.nn as nn
import numpy as np
import argparse
import os
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import custom modules
from data import get_mnist, get_cifar10, get_cifar100, FederatedSampler
from models import get_model, LocalUpdate
from auditor import (
    Generator, Discriminator, train_generator, generate_synthetic_samples,
    FairnessProbeGenerator, FairnessAuditor, FairnessContributionScorer, ClientScorer
)
from auditor.utils.scoring import compute_client_update, aggregate_weighted
from utils import test_model, save_results, plot_results

# WandB for experiment tracking (optional)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Fed-AuditGAN: Fairness-Aware Federated Learning')
    
    # Dataset and model
    parser.add_argument('--dataset', type=str, default='mnist', 
                       choices=['mnist', 'cifar10', 'cifar100'],
                       help='Dataset to use')
    parser.add_argument('--data_root', type=str, default='../datasets/',
                       help='Root directory for datasets')
    parser.add_argument('--model_name', type=str, default='cnn',
                       choices=['cnn', 'mlp'],
                       help='Model architecture')
    
    # Federated learning settings
    parser.add_argument('--n_clients', type=int, default=10,
                       help='Number of federated clients')
    parser.add_argument('--n_epochs', type=int, default=50,
                       help='Number of federated rounds')
    parser.add_argument('--n_client_epochs', type=int, default=5,
                       help='Number of local training epochs per client')
    parser.add_argument('--batch_size', type=int, default=10,
                       help='Local training batch size')
    parser.add_argument('--lr', type=float, default=0.01,
                       help='Learning rate for local training')
    parser.add_argument('--frac', type=float, default=1.0,
                       help='Fraction of clients to sample per round')
    
    # Data partitioning
    parser.add_argument('--partition_mode', type=str, default='shard',
                       choices=['iid', 'shard', 'dirichlet'],
                       help='Data partitioning strategy')
    parser.add_argument('--n_shards', type=int, default=200,
                       help='Number of shards for shard-based partitioning')
    parser.add_argument('--dirichlet_alpha', type=float, default=0.1,
                       help='Alpha parameter for Dirichlet partitioning')
    
    # Fed-AuditGAN specific settings
    parser.add_argument('--use_audit_gan', action='store_true',
                       help='Enable Fed-AuditGAN fairness auditing with DCGAN')
    parser.add_argument('--gamma', type=float, default=0.5,
                       help='Balance between fairness and accuracy [0-1]. 0=pure accuracy, 1=pure fairness')
    parser.add_argument('--n_audit_steps', type=int, default=50,
                       help='DCGAN training epochs per round (default: 50)')
    parser.add_argument('--n_probes', type=int, default=1000,
                       help='Number of synthetic fairness probes to generate')
    parser.add_argument('--latent_dim', type=int, default=100,
                       help='Latent dimension for DCGAN generator')
    parser.add_argument('--use_legacy_generator', action='store_true',
                       help='Use legacy autoencoder-based generator instead of DCGAN')
    parser.add_argument('--sensitive_attrs', nargs='+', type=int, default=None,
                       help='Indices of sensitive attributes (for legacy generator only)')
    
    # Experiment tracking
    parser.add_argument('--wandb', action='store_true',
                       help='Use Weights & Biases for logging')
    parser.add_argument('--exp_name', type=str, default=None,
                       help='Experiment name for W&B')
    parser.add_argument('--save_dir', type=str, default='./results',
                       help='Directory to save results')
    
    # System
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device for computation')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Initialize W&B if requested
    if args.wandb and WANDB_AVAILABLE:
        wandb.init(
            project="fed-audit-gan",
            name=args.exp_name or f"{args.dataset}_{args.partition_mode}_gamma{args.gamma}",
            config=vars(args)
        )
    elif args.wandb:
        print("⚠ Warning: wandb not installed. Continuing without logging.")
    
    # Print configuration
    print_header("Fed-AuditGAN Configuration")
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model_name}")
    print(f"Partition Mode: {args.partition_mode}")
    print(f"Clients: {args.n_clients}")
    print(f"Rounds: {args.n_epochs}")
    print(f"Fed-AuditGAN: {'ENABLED' if args.use_audit_gan else 'DISABLED'}")
    if args.use_audit_gan:
        print(f"Gamma (fairness weight): {args.gamma}")
    print(f"Device: {args.device}\n")
    
    # Load dataset
    print_header("Loading Dataset")
    if args.dataset == 'mnist':
        train_dataset, test_dataset = get_mnist(args.data_root)
        num_classes = 10
        input_channels = 1
    elif args.dataset == 'cifar10':
        train_dataset, test_dataset = get_cifar10(args.data_root)
        num_classes = 10
        input_channels = 3
    elif args.dataset == 'cifar100':
        train_dataset, test_dataset = get_cifar100(args.data_root)
        num_classes = 100
        input_channels = 3
    
    # Partition data among clients
    sampler = FederatedSampler(
        dataset=train_dataset,
        n_clients=args.n_clients,
        partition_mode=args.partition_mode,
        n_shards=args.n_shards,
        dirichlet_alpha=args.dirichlet_alpha,
        seed=args.seed
    )
    
    # Create test loader
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=64, shuffle=False
    )
    
    # Create validation loader (subset of training data for server)
    val_indices = np.random.choice(len(train_dataset), size=1000, replace=False)
    val_dataset = torch.utils.data.Subset(train_dataset, val_indices)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=32, shuffle=True
    )
    
    # Initialize global model
    print_header("Initializing Global Model")
    global_model = get_model(args.model_name, args.dataset, num_classes)
    global_model.to(args.device)
    print(f"Model: {args.model_name}")
    print(f"Parameters: {sum(p.numel() for p in global_model.parameters()):,}\n")
    
    # Training history
    history = {
        'train_loss': [],
        'test_accuracy': [],
        'fairness_scores': [],
        'accuracy_scores': [],
        'bias_scores': []
    }
    
    # =========================================================================
    # MAIN TRAINING LOOP
    # =========================================================================
    
    for round_idx in range(args.n_epochs):
        print_header(f"Round {round_idx + 1}/{args.n_epochs}")
        
        # Phase 1: Standard FL Training Round
        # ====================================
        print("\n[Phase 1] Client Training")
        
        # Sample clients for this round
        n_sampled = max(int(args.frac * args.n_clients), 1)
        sampled_clients = np.random.choice(
            args.n_clients, n_sampled, replace=False
        ).tolist()
        print(f"Sampled {n_sampled} clients: {sampled_clients}")
        
        # Store client updates and losses
        client_updates = []
        client_losses = []
        
        # Train each client
        pbar = tqdm(sampled_clients, desc="Training clients")
        for client_id in pbar:
            # Get client's data
            client_loader = sampler.get_client_loader(
                client_id, batch_size=args.batch_size
            )
            
            # Create local updater
            local_updater = LocalUpdate(
                dataset=sampler.dataset,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                local_epochs=args.n_client_epochs,
                device=args.device
            )
            local_updater.trainloader = client_loader
            
            # Copy global model for local training
            local_model = copy.deepcopy(global_model)
            model_before = copy.deepcopy(global_model)
            
            # Train locally
            local_model, losses = local_updater.train(local_model)
            
            # Compute update: Δ = M_local - M_global
            update = compute_client_update(model_before, local_model)
            client_updates.append(update)
            client_losses.append(np.mean(losses))
            
            pbar.set_postfix({'loss': f"{np.mean(losses):.4f}"})
        
        avg_train_loss = np.mean(client_losses)
        history['train_loss'].append(avg_train_loss)
        print(f"✓ Phase 1 complete. Avg loss: {avg_train_loss:.4f}")
        
        # Phase 2, 3, 4: Fed-AuditGAN or Standard Aggregation
        # ====================================================
        
        if args.use_audit_gan:
            # Phase 2: Generative Fairness Auditing
            # =======================================
            
            if args.use_legacy_generator:
                # Legacy autoencoder-based approach
                print("\n[Phase 2] Generative Fairness Auditing (Legacy Autoencoder)")
                
                # Determine input dimension
                if args.dataset == 'mnist':
                    input_dim = 28 * 28
                else:  # CIFAR
                    input_dim = 32 * 32 * 3
                
                legacy_generator = FairnessProbeGenerator(
                    input_dim=input_dim,
                    hidden_dims=[256, 128, 64],
                    sensitive_attrs_indices=args.sensitive_attrs
                )
                
                # Create auditor with legacy approach
                legacy_auditor = FairnessAuditor(legacy_generator, global_model, args.device)
                
                # Train generator
                legacy_auditor.train_generator(
                    seed_data_loader=val_loader,
                    n_steps=args.n_audit_steps,
                    alpha=1.0,
                    beta=0.5
                )
                
                # Generate probes
                probes = legacy_auditor.generate_probes(val_loader, n_probes=args.n_probes)
                probe_loader = torch.utils.data.DataLoader(probes, batch_size=32)
                
                print(f"✓ Phase 2 complete (legacy mode).")
                
            else:
                # DCGAN-based approach (default)
                print("\n[Phase 2] Generative Fairness Auditing (DCGAN)")
                
                # Determine image shape for DCGAN
                if args.dataset == 'mnist':
                    img_shape = (1, 28, 28)
                else:  # CIFAR
                    img_shape = (3, 32, 32)
                
                # Initialize DCGAN Generator and Discriminator
                generator = Generator(
                    latent_dim=args.latent_dim,
                    num_classes=num_classes,
                    img_shape=img_shape
                )
                discriminator = Discriminator(
                    num_classes=num_classes,
                    img_shape=img_shape
                )
                
                print(f"Training DCGAN for {args.n_audit_steps} epochs...")
                
                # Train DCGAN to generate synthetic fairness probes
                generator, discriminator = train_generator(
                    generator=generator,
                    discriminator=discriminator,
                    dataloader=val_loader,
                    n_epochs=args.n_audit_steps,
                    device=args.device,
                    lr=0.0002,
                    sample_interval=max(args.n_audit_steps // 5, 1)
                )
                
                # Generate synthetic samples for fairness auditing
                print(f"Generating {args.n_probes} synthetic samples...")
                synthetic_imgs, synthetic_labels = generate_synthetic_samples(
                    generator=generator,
                    num_samples=args.n_probes,
                    device=args.device
                )
                
                # Create probe dataset
                probe_dataset = torch.utils.data.TensorDataset(
                    synthetic_imgs.cpu(), synthetic_labels.cpu()
                )
                probe_loader = torch.utils.data.DataLoader(
                    probe_dataset, batch_size=32, shuffle=False
                )
            
            # Create fairness auditor
            auditor = FairnessAuditor(
                num_classes=num_classes,
                device=args.device
            )
            
            # Audit global model fairness
            fairness_metrics = auditor.audit_model(
                model=global_model,
                dataloader=probe_loader
            )
            
            # Store baseline fairness metrics
            baseline_bias = fairness_metrics['demographic_parity']
            history['bias_scores'].append(baseline_bias)
            
            print(f"✓ Phase 2 complete.")
            print(f"  Demographic Parity: {fairness_metrics['demographic_parity']:.6f}")
            print(f"  Equalized Odds: {fairness_metrics['equalized_odds']:.6f}")
            print(f"  Class Balance: {fairness_metrics['class_balance']:.6f}")
            
            # Phase 3: Fairness Contribution Scoring
            # =======================================
            print("\n[Phase 3] Fairness Contribution Scoring")
            
            # Evaluate each client's contribution
            client_accuracies = []
            client_fairness_metrics = []
            
            print("Evaluating client contributions...")
            for idx, update in enumerate(client_updates):
                # Create hypothetical model with client's update
                hypothetical_model = copy.deepcopy(global_model)
                hypothetical_dict = hypothetical_model.state_dict()
                
                # Apply update: M_new = M_global + Δ
                for key in hypothetical_dict.keys():
                    if key in update:
                        hypothetical_dict[key] = hypothetical_dict[key] + update[key]
                
                hypothetical_model.load_state_dict(hypothetical_dict)
                
                # Measure accuracy on validation set
                hypothetical_model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for data, target in val_loader:
                        data, target = data.to(args.device), target.to(args.device)
                        output = hypothetical_model(data)
                        pred = output.argmax(dim=1)
                        correct += (pred == target).sum().item()
                        total += target.size(0)
                
                client_acc = correct / total if total > 0 else 0.0
                client_accuracies.append(client_acc)
                
                # Measure fairness on synthetic probes
                client_fairness = auditor.audit_model(
                    model=hypothetical_model,
                    dataloader=probe_loader
                )
                client_fairness_metrics.append(client_fairness)
                
                print(f"  Client {idx}: Acc={client_acc:.4f}, "
                      f"DP={client_fairness['demographic_parity']:.4f}")
            
            # Compute global accuracy and fairness
            global_model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(args.device), target.to(args.device)
                    output = global_model(data)
                    pred = output.argmax(dim=1)
                    correct += (pred == target).sum().item()
                    total += target.size(0)
            
            global_accuracy = correct / total if total > 0 else 0.0
            
            # Use FairnessContributionScorer to compute weights with gamma-based weighting
            alpha = 1.0 - args.gamma  # Accuracy weight
            beta = args.gamma          # Fairness weight
            
            logger.info(f"Computing contribution scores with gamma={args.gamma:.2f}")
            logger.info(f"  → Accuracy weight (alpha): {alpha:.2f}")
            logger.info(f"  → Fairness weight (beta):  {beta:.2f}")
            
            scorer = FairnessContributionScorer(
                alpha=alpha,
                beta=beta
            )
            
            final_weights = scorer.compute_combined_scores(
                client_accuracies=client_accuracies,
                global_accuracy=global_accuracy,
                client_fairness_scores=client_fairness_metrics,
                global_fairness_score=fairness_metrics
            )
            
            logger.info(f"✓ Phase 3 complete. Client weights computed:")
            logger.info(f"  → Weights: {[f'{w:.4f}' for w in final_weights]}")
            logger.info(f"  → Max: {max(final_weights):.4f}, Min: {min(final_weights):.4f}")
            logger.info(f"  → Std Dev: {np.std(final_weights):.4f}")
            logger.info(f"  → Weight variance indicates fairness-driven reweighting")
            
            # Store statistics
            avg_fairness = np.mean([
                m['demographic_parity'] for m in client_fairness_metrics
            ])
            history['fairness_scores'].append(avg_fairness)
            history['accuracy_scores'].append(np.mean(client_accuracies))
            
            print(f"✓ Phase 3 complete.")
            print(f"  Avg Client Accuracy: {np.mean(client_accuracies):.4f}")
            print(f"  Avg Client Fairness: {avg_fairness:.4f}")
            print(f"  Weights: {[f'{w:.3f}' for w in final_weights]}")
            
            # Phase 4: Multi-Objective Aggregation
            # =====================================
            print("\n[Phase 4] Multi-Objective Aggregation")
            
            global_model = aggregate_weighted(
                global_model, client_updates, final_weights
            )
            
            print(f"✓ Phase 4 complete. Model updated with fairness-aware weights.")
            
        else:
            # Standard FedAvg aggregation
            print("\n[Aggregation] Standard FedAvg")
            
            # Uniform weights
            uniform_weights = [1.0 / len(client_updates)] * len(client_updates)
            global_model = aggregate_weighted(
                global_model, client_updates, uniform_weights
            )
            
            print(f"✓ Aggregation complete.")
        
        # Evaluate global model
        # =====================
        test_acc, test_loss = test_model(global_model, test_loader, args.device)
        history['test_accuracy'].append(test_acc)
        
        print(f"\n{'='*60}")
        print(f"Round {round_idx + 1} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Test Accuracy: {test_acc:.2f}%")
        print(f"  Test Loss: {test_loss:.4f}")
        if args.use_audit_gan:
            print(f"  Baseline Bias: {baseline_bias:.6f}")
            print(f"  Avg Fairness Score: {history['fairness_scores'][-1]:.6f}")
            print(f"  Avg Accuracy Score: {history['accuracy_scores'][-1]:.6f}")
        print(f"{'='*60}\n")
        
        # Log to W&B
        if args.wandb and WANDB_AVAILABLE:
            log_dict = {
                'round': round_idx + 1,
                'train_loss': avg_train_loss,
                'test_accuracy': test_acc,
                'test_loss': test_loss
            }
            if args.use_audit_gan:
                log_dict.update({
                    'baseline_bias': baseline_bias,
                    'avg_fairness_score': history['fairness_scores'][-1],
                    'avg_accuracy_score': history['accuracy_scores'][-1]
                })
            wandb.log(log_dict)
    
    # =========================================================================
    # TRAINING COMPLETE
    # =========================================================================
    
    print_header("Training Complete!")
    print(f"Final Test Accuracy: {history['test_accuracy'][-1]:.2f}%")
    if args.use_audit_gan:
        print(f"Final Bias Score: {history['bias_scores'][-1]:.6f}")
    
    # Save results
    os.makedirs(args.save_dir, exist_ok=True)
    save_results(history, args, os.path.join(args.save_dir, 'history.pkl'))
    
    # Plot results
    plot_results(history, args, args.save_dir)
    
    # Save model
    model_path = os.path.join(args.save_dir, 'final_model.pth')
    torch.save(global_model.state_dict(), model_path)
    print(f"\n✓ Model saved to {model_path}")
    
    if args.wandb and WANDB_AVAILABLE:
        wandb.finish()


def print_header(text):
    """Print formatted header."""
    print(f"\n{'='*60}")
    print(f"{text:^60}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
