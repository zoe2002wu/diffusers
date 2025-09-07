from diffusers import StableDiffusionPipeline
import torch

# Load the pipeline
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")

# Generate samples for FID evaluation
def generate_samples_for_fid(num_samples=50000, guidance_scale=7.5):
    all_images = []
    
    for i in range(0, num_samples, batch_size):
        # Generate unconditional samples (no prompt)
        with torch.no_grad():
            images = pipe(
                prompt="",  # Empty prompt for unconditional generation
                num_images_per_prompt=batch_size,
                guidance_scale=guidance_scale,
                num_inference_steps=50
            ).images
        
        # Convert to numpy arrays
        for img in images:
            img_array = np.array(img)  # Convert PIL to numpy
            all_images.append(img_array)
    
    return np.array(all_images)