import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

# Set random seeds for reproducibility
random_seed = 42
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Custom Dataset class for paired images (real person -> anime style)
class AnimeDataset(Dataset):
    def __init__(self, real_dir, anime_dir, transform=None):
        """
        Args:
            real_dir (string): Directory with real person images
            anime_dir (string): Directory with corresponding anime style images
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.real_dir = real_dir
        self.anime_dir = anime_dir
        self.transform = transform
        
        # Get all image filenames (assuming they have the same names in both directories)
        self.filenames = [f for f in os.listdir(real_dir) if os.path.isfile(os.path.join(real_dir, f)) and
                         f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        real_img_path = os.path.join(self.real_dir, self.filenames[idx])
        anime_img_path = os.path.join(self.anime_dir, self.filenames[idx])
        
        real_image = Image.open(real_img_path).convert('RGB')
        anime_image = Image.open(anime_img_path).convert('RGB')
        
        if self.transform:
            real_image = self.transform(real_image)
            anime_image = self.transform(anime_image)
            
        return {'real': real_image, 'anime': anime_image}

# Generator architecture (U-Net style)
class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(Generator, self).__init__()
        
        # Encoder (downsampling)
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # Decoder (upsampling)
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True)
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(1024, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.dec5 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Final layer
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
        
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        
        # Bottleneck
        b = self.bottleneck(e5)
        
        # Decoder with skip connections
        d1 = self.dec1(b)
        d2 = self.dec2(torch.cat([d1, e5], 1))
        d3 = self.dec3(torch.cat([d2, e4], 1))
        d4 = self.dec4(torch.cat([d3, e3], 1))
        d5 = self.dec5(torch.cat([d4, e2], 1))
        
        # Final
        out = self.final(torch.cat([d5, e1], 1))
        
        return out

# Discriminator architecture (PatchGAN)
class Discriminator(nn.Module):
    def __init__(self, in_channels=6):  # Takes both real and generated images
        super(Discriminator, self).__init__()
        
        # A series of convolutional layers
        self.model = nn.Sequential(
            # Layer 1
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 2
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 3
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 4
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 5 (output layer)
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        )
        
    def forward(self, real_img, generated_img):
        # Concatenate the real and generated images
        x = torch.cat([real_img, generated_img], dim=1)
        return self.model(x)

# Initialize weights function
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Training function
def train_anime_gan(real_dir, anime_dir, output_dir, 
                   batch_size=4, num_epochs=100, 
                   lr=0.0002, beta1=0.5, 
                   image_size=256, 
                   save_interval=10):
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Create dataset and dataloader
    dataset = AnimeDataset(real_dir, anime_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # Initialize models
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    
    # Apply weight initialization
    generator.apply(weights_init)
    discriminator.apply(weights_init)
    
    # Loss functions
    criterion_gan = nn.BCEWithLogitsLoss()
    criterion_pixel = nn.L1Loss()
    
    # Lambda for L1 loss
    lambda_pixel = 100
    
    # Setup optimizers
    optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
    
    # Training loop
    for epoch in range(num_epochs):
        loop = tqdm(dataloader, leave=True)
        
        for batch_idx, batch in enumerate(loop):
            # Get batch data
            real_images = batch['real'].to(device)
            anime_images = batch['anime'].to(device)
            
            # Create target labels (real = 1, fake = 0)
            real_label = torch.ones(real_images.size(0), 1, 30, 30).to(device)  # Size depends on discriminator output
            fake_label = torch.zeros(real_images.size(0), 1, 30, 30).to(device)
            
            # -----------------------
            # Train Generator
            # -----------------------
            optimizer_g.zero_grad()
            
            # Generate anime-style images
            fake_anime = generator(real_images)
            
            # Compute GAN loss
            pred_fake = discriminator(real_images, fake_anime)
            loss_gan = criterion_gan(pred_fake, real_label)
            
            # Compute pixel-wise loss
            loss_pixel = criterion_pixel(fake_anime, anime_images)
            
            # Total Generator loss
            loss_g = loss_gan + lambda_pixel * loss_pixel
            
            # Backward pass and optimize
            loss_g.backward()
            optimizer_g.step()
            
            # -----------------------
            # Train Discriminator
            # -----------------------
            optimizer_d.zero_grad()
            
            # Real loss
            pred_real = discriminator(real_images, anime_images)
            loss_real = criterion_gan(pred_real, real_label)
            
            # Fake loss
            pred_fake = discriminator(real_images, fake_anime.detach())
            loss_fake = criterion_gan(pred_fake, fake_label)
            
            # Total Discriminator loss
            loss_d = (loss_real + loss_fake) / 2
            
            # Backward pass and optimize
            loss_d.backward()
            optimizer_d.step()
            
            # Update progress bar
            loop.set_postfix(
                d_loss=f"{loss_d.item():.4f}",
                g_loss=f"{loss_g.item():.4f}",
                epoch=f"{epoch+1}/{num_epochs}"
            )
        
        # Save models and example outputs at intervals
        if (epoch+1) % save_interval == 0 or epoch == num_epochs-1:
            # Save models
            torch.save(generator.state_dict(), f"{output_dir}/generator_epoch_{epoch+1}.pth")
            torch.save(discriminator.state_dict(), f"{output_dir}/discriminator_epoch_{epoch+1}.pth")
            
            # Save sample images
            with torch.no_grad():
                # Get a batch of test images
                test_real = next(iter(dataloader))['real'].to(device)
                test_anime = next(iter(dataloader))['anime'].to(device)
                
                # Generate fake anime images
                fake_anime = generator(test_real)
                
                # Convert to CPU and denormalize
                test_real = test_real.cpu() * 0.5 + 0.5
                test_anime = test_anime.cpu() * 0.5 + 0.5
                fake_anime = fake_anime.cpu() * 0.5 + 0.5
                
                # Create and save a grid of images
                fig, axes = plt.subplots(3, min(4, batch_size), figsize=(15, 10))
                
                for i in range(min(4, batch_size)):
                    # Plot real person
                    axes[0, i].imshow(test_real[i].permute(1, 2, 0))
                    axes[0, i].set_title("Real Person")
                    axes[0, i].axis("off")
                    
                    # Plot ground truth anime
                    axes[1, i].imshow(test_anime[i].permute(1, 2, 0))
                    axes[1, i].set_title("Ground Truth Anime")
                    axes[1, i].axis("off")
                    
                    # Plot generated anime
                    axes[2, i].imshow(fake_anime[i].permute(1, 2, 0))
                    axes[2, i].set_title("Generated Anime")
                    axes[2, i].axis("off")
                
                plt.tight_layout()
                plt.savefig(f"{output_dir}/samples_epoch_{epoch+1}.png")
                plt.close()
    
    print("Training completed!")
    return generator, discriminator

# Inference function to generate anime-style images from real photos
def generate_anime(generator_path, input_dir, output_dir, image_size=256):
    """
    Generate anime-style images from real photos using a trained generator
    
    Args:
        generator_path: Path to the trained generator model
        input_dir: Directory containing input real photos
        output_dir: Directory to save generated anime-style images
        image_size: Size to resize images to
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the generator
    generator = Generator().to(device)
    generator.load_state_dict(torch.load(generator_path, map_location=device))
    generator.eval()
    
    # Define transforms for inference
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Get list of image files
    image_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and
                  f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"Processing {len(image_files)} images...")
    
    # Process each image
    for img_file in tqdm(image_files):
        # Load and preprocess image
        img_path = os.path.join(input_dir, img_file)
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        # Generate anime-style image
        with torch.no_grad():
            generated = generator(img_tensor)
            
            # Convert to image and denormalize
            generated = generated.cpu().squeeze(0) * 0.5 + 0.5
            generated_img = transforms.ToPILImage()(generated)
            
            # Save the generated image
            output_path = os.path.join(output_dir, f"anime_{img_file}")
            generated_img.save(output_path)
    
    print(f"Generated anime-style images saved to {output_dir}")

# Example usage
if __name__ == "__main__":
    # Define your directories
    real_dir = "path/to/your/real_photos"
    anime_dir = "path/to/your/anime_style_photos"
    output_dir = "path/to/save/models_and_samples"
    
    # Train the model
    generator, discriminator = train_anime_gan(
        real_dir=real_dir,
        anime_dir=anime_dir,
        output_dir=output_dir,
        batch_size=4,
        num_epochs=200,
        lr=0.0002,
        beta1=0.5,
        image_size=256,
        save_interval=10
    )
    
    # After training, you can use the generator for inference
    generate_anime(
        generator_path=f"{output_dir}/generator_epoch_200.pth",
        input_dir="path/to/test_photos",
        output_dir="path/to/save/generated_anime",
        image_size=256
    )