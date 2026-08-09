import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch_optimizer as optim # The LARS optimizer package
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

from SimCLR import SimCLR
from NT_Xent import NTXentLoss
from augmentation import SimCLRAugmentation

def main():
    # 1. Paper-Accurate Hyperparameters Adapted for RTX 5090
    batch_size = 128  
    epochs = 2000     # Increased to compensate for the smaller batch size
    temperature = 0.5
    weight_decay = 1e-6
    
    # The paper's exact LR scaling rule: 0.3 * (BatchSize / 256)
    base_lr = 0.3 * (batch_size / 256) 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Data Loading
    print("Loading dataset...")
    data_dir = "Faces" 
    
    augmentations = SimCLRAugmentation(image_size=224)
    dataset = ImageFolder(root=data_dir, transform=augmentations)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                            num_workers=8, pin_memory=True, drop_last=True)

# 3. Model & Loss Setup
    print("Initializing Model and Loss...")
    model = SimCLR(projection_dim=128).to(device)
    criterion = NTXentLoss(temperature=temperature).to(device)

    # 4. Optimizer: LARS with Weight Decay Exclusion
    print("Configuring LARS Optimizer and Parameter Groups...")
    
    # Helper function to identify bias and batch norm layers
    def exclude_from_weight_decay(name):
        return "bias" in name or "bn" in name or "batchnorm" in name

    # Separate parameters into two groups
    regular_params = []
    excluded_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if exclude_from_weight_decay(name):
            excluded_params.append(param)
        else:
            regular_params.append(param)

    # Pass the groups to the optimizer: regular params get decay, excluded get 0.0
    param_groups = [
        {"params": regular_params, "weight_decay": weight_decay},
        {"params": excluded_params, "weight_decay": 0.0}
    ]

    optimizer = optim.LARS(
        param_groups, 
        lr=base_lr, 
        momentum=0.9
    )

    # 5. The Scheduler: Linear Warmup (10 epochs) -> Cosine Decay (1990 epochs)
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=10)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=(epochs - 10))
    
    # Chain them together
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[10])

    # 6. The Training Loop
    print(f"Starting Training on {device} with LR: {base_lr} for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for (view_1, view_2), _ in pbar:
            view_1 = view_1.to(device)
            view_2 = view_2.to(device)

            optimizer.zero_grad()

            _, z_1 = model(view_1)
            _, z_2 = model(view_2)

            loss = criterion(z_1, z_2)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
            # Show current learning rate and loss in the progress bar
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({"Loss": loss.item(), "LR": f"{current_lr:.4f}"})

        # Step the chained scheduler at the end of every epoch
        scheduler.step()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}] Completed | Average Loss: {avg_loss:.4f}")

        # Save a checkpoint exactly every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = f"simclr_checkpoint_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"--> Saved checkpoint: {checkpoint_path}")

if __name__ == "__main__":
    main()