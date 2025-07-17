import click
import torch as t
import numpy as np
from PIL import Image

from oprl.environment import make_env


def create_webp_gif(numpy_arrays, output_path, duration=100, loop=0):
    """
    Create a WebP animated image from a list of NumPy arrays.
    
    Args:
        numpy_arrays: List of NumPy arrays (each representing a frame)
        output_path: Output file path (should end with .webp)
        duration: Duration between frames in milliseconds
        loop: Number of loops (0 = infinite loop)
    """
    # Convert NumPy arrays to PIL Images
    pil_images = []
    
    for arr in numpy_arrays:
        # Ensure the array is in the right format
        if arr.dtype != np.uint8:
            # Normalize to 0-255 range if needed
            if arr.max() <= 1.0:
                arr = (arr * 255).astype(np.uint8)
            else:
                arr = arr.astype(np.uint8)
        
        # Handle different array shapes
        if len(arr.shape) == 2:  # Grayscale
            img = Image.fromarray(arr, mode='L')
        elif len(arr.shape) == 3:  # RGB/RGBA
            if arr.shape[2] == 3:
                img = Image.fromarray(arr, mode='RGB')
            elif arr.shape[2] == 4:
                img = Image.fromarray(arr, mode='RGBA')
            else:
                raise ValueError(f"Unsupported number of channels: {arr.shape[2]}")
        else:
            raise ValueError(f"Unsupported array shape: {arr.shape}")
        
        pil_images.append(img)
    
    # Save as animated WebP
    pil_images[0].save(
        output_path,
        format='WebP',
        save_all=True,
        append_images=pil_images[1:],
        duration=duration,
        loop=loop,
        optimize=True
    )


@click.command()
@click.option("--policy", "-p", help="Path to policy weights.")
@click.option("--output", "-o", default="policy.webp", help="Path to output file.")
@click.option("--env", "-e", default="walker-walk", help="Environemnt name.")
@click.option("--seed", "-s", default=0, help="Environment seed.")
def visualize_policy(policy, output, env, seed):
    env = make_env(env, seed=seed)

    actor = t.load(policy, weights_only=False)
    print("Actor loaded: ", type(actor))

    imgs = []
    state, _ = env.reset()
    done = False
    while not done:
        img = np.expand_dims(env.render(), axis=0)  # [1, W, H, C]
        imgs.append(img)
        action = actor.exploit(t.from_numpy(state))
        state, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    print("imgs: ", len(imgs), imgs[0].shape)
    frames = np.concatenate(imgs, dtype="uint8", axis=0)
    print("frames: ", frames.shape)

    # Create the WebP GIF
    create_webp_gif(frames, output, duration=25)
    print("WebP GIF for dm_control created successfully!")


if __name__ == "__main__":
    visualize_policy()
