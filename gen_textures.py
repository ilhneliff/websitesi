"""Generate high-quality cloud and star textures using Pillow + numpy."""
import numpy as np
from PIL import Image, ImageFilter
import os

OUT = "static/textures"


def perlin_like(w, h, scale=4, octaves=6, seed=42):
    """Multi-octave value noise."""
    rng = np.random.RandomState(seed)
    result = np.zeros((h, w), dtype=np.float64)
    amp = 1.0
    freq = scale
    for _ in range(octaves):
        # Generate random grid
        gw = int(freq) + 2
        gh = int(freq * h / w) + 2
        grid = rng.rand(gh, gw)

        # Upsample with bicubic
        grid_img = Image.fromarray((grid * 255).astype(np.uint8), 'L')
        grid_img = grid_img.resize((w, h), Image.BICUBIC)
        layer = np.array(grid_img, dtype=np.float64) / 255.0

        result += layer * amp
        amp *= 0.5
        freq *= 2.0

    # Normalize
    result = (result - result.min()) / (result.max() - result.min() + 1e-10)
    return result


def make_cloud_texture(filename, w=1024, h=512, seed=42):
    """Volumetric-looking cloud with smooth transparent edges."""
    noise1 = perlin_like(w, h, scale=3, octaves=6, seed=seed)
    noise2 = perlin_like(w, h, scale=5, octaves=5, seed=seed + 100)
    noise3 = perlin_like(w, h, scale=8, octaves=4, seed=seed + 200)

    # Elliptical falloff mask
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = w / 2, h / 2
    dx = (xx - cx) / (w * 0.4)
    dy = (yy - cy) / (h * 0.35)
    dist = np.sqrt(dx ** 2 + dy ** 2)
    mask = np.clip(1.0 - dist, 0, 1)
    mask = mask ** 1.5  # softer falloff

    # Combine noise with mask for fluffy shape
    density = mask * (noise1 * 0.5 + noise2 * 0.3 + 0.2)
    density = np.clip((density - 0.15) * 2.0, 0, 1)
    density = density ** 0.8  # soften

    # Internal detail variation
    detail = noise3 * 0.3 + 0.7
    density *= detail

    # Color: blue-tinted white with top-lighting
    y_norm = yy / h
    top_light = 1.0 - y_norm * 0.25

    r_base = (0.68 + density * 0.3) * top_light
    g_base = (0.72 + density * 0.25) * top_light
    b_base = (0.82 + density * 0.18) * top_light

    r = np.clip(r_base * 255, 0, 255).astype(np.uint8)
    g = np.clip(g_base * 255, 0, 255).astype(np.uint8)
    b = np.clip(b_base * 255, 0, 255).astype(np.uint8)
    a = np.clip(density * 230, 0, 255).astype(np.uint8)

    img = np.stack([r, g, b, a], axis=2)
    result = Image.fromarray(img, 'RGBA')

    # Apply gaussian blur for softness
    result = result.filter(ImageFilter.GaussianBlur(radius=4))

    result.save(os.path.join(OUT, filename), 'PNG')
    print(f"  -> {filename} ({w}x{h})")


def make_star_texture(filename, size=128):
    """Soft glowing star particle with smooth falloff."""
    img = np.zeros((size, size, 4), dtype=np.uint8)
    center = size / 2
    for y in range(size):
        for x in range(size):
            dx = (x - center) / center
            dy = (y - center) / center
            d = np.sqrt(dx * dx + dy * dy)

            # Core: bright sharp center
            core = np.exp(-d * d * 12) * 1.0
            # Inner glow
            inner = np.exp(-d * d * 3) * 0.6
            # Outer soft glow
            outer = np.exp(-d * d * 0.8) * 0.2

            brightness = min(1.0, core + inner + outer)
            alpha = min(1.0, (core + inner + outer * 0.5))

            # Slightly warm white color
            img[y, x, 0] = int(min(255, brightness * 245 + 10))
            img[y, x, 1] = int(min(255, brightness * 240 + 10))
            img[y, x, 2] = int(min(255, brightness * 255))
            img[y, x, 3] = int(alpha * 255)

    result = Image.fromarray(img, 'RGBA')
    result.save(os.path.join(OUT, filename), 'PNG')
    print(f"  -> {filename} ({size}x{size})")


def make_star_flare(filename, size=256):
    """Star with cross-flare effect for bright stars."""
    img = np.zeros((size, size, 4), dtype=np.float64)
    center = size / 2

    for y in range(size):
        for x in range(size):
            dx = (x - center) / center
            dy = (y - center) / center
            d = np.sqrt(dx * dx + dy * dy)

            # Radial glow
            glow = np.exp(-d * d * 6)
            core = np.exp(-d * d * 25)

            # Cross flare (horizontal + vertical)
            flare_h = np.exp(-dy * dy * 80) * np.exp(-abs(dx) * 3) * 0.4
            flare_v = np.exp(-dx * dx * 80) * np.exp(-abs(dy) * 3) * 0.4

            # Diagonal flares
            d1 = abs(dx - dy) / 1.414
            d2 = abs(dx + dy) / 1.414
            flare_d1 = np.exp(-d1 * d1 * 120) * np.exp(-d * 4) * 0.15
            flare_d2 = np.exp(-d2 * d2 * 120) * np.exp(-d * 4) * 0.15

            total = core + glow * 0.5 + flare_h + flare_v + flare_d1 + flare_d2

            img[y, x, 0] = min(1.0, total * 0.95)
            img[y, x, 1] = min(1.0, total * 0.93)
            img[y, x, 2] = min(1.0, total)
            img[y, x, 3] = min(1.0, total)

    result = Image.fromarray((img * 255).astype(np.uint8), 'RGBA')
    result.save(os.path.join(OUT, filename), 'PNG')
    print(f"  -> {filename} ({size}x{size})")


def make_moon_glow(filename, size=512):
    """Soft radial glow texture for moon halo."""
    img = np.zeros((size, size, 4), dtype=np.float64)
    center = size / 2

    for y in range(size):
        for x in range(size):
            dx = (x - center) / center
            dy = (y - center) / center
            d = np.sqrt(dx * dx + dy * dy)

            # Multi-layer glow
            g1 = np.exp(-d * d * 1.5) * 0.4
            g2 = np.exp(-d * d * 4) * 0.3
            g3 = np.exp(-d * d * 12) * 0.3

            total = g1 + g2 + g3

            # Slightly blue-white
            img[y, x, 0] = min(1.0, total * 0.85)
            img[y, x, 1] = min(1.0, total * 0.9)
            img[y, x, 2] = min(1.0, total)
            img[y, x, 3] = min(1.0, total)

    result = Image.fromarray((img * 255).astype(np.uint8), 'RGBA')
    result = result.filter(ImageFilter.GaussianBlur(radius=2))
    result.save(os.path.join(OUT, filename), 'PNG')
    print(f"  -> {filename} ({size}x{size})")


if __name__ == "__main__":
    print("Generating textures...")

    # Cloud textures (4 variants)
    print("\nClouds:")
    make_cloud_texture("cloud_a.png", 1024, 512, seed=42)
    make_cloud_texture("cloud_b.png", 1024, 512, seed=99)
    make_cloud_texture("cloud_c.png", 1024, 512, seed=7)
    make_cloud_texture("cloud_d.png", 1024, 512, seed=55)

    # Star textures
    print("\nStars:")
    make_star_texture("star_soft.png", 128)
    make_star_flare("star_flare.png", 256)

    # Moon glow
    print("\nMoon glow:")
    make_moon_glow("moon_glow.png", 512)

    print("\nDone! All textures saved to", OUT)
