# Pixelization

[![Crates.io](https://img.shields.io/crates/v/pixelization.svg)](https://crates.io/crates/pixelization)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Pixelization** is a high-performance Rust library and CLI tool for transforming images into pixel art. 

It implements advanced quantization algorithms, including:
1.  **K-Means Clustering:** A classic statistical approach to find the best color palette.
2.  **Pixelated Image Abstraction (PIA):** An implementation of the paper [Pixelated Image Abstraction (Gerstner et al., 2012)](https://dl.acm.org/doi/10.5555/2330147.2330154), which utilizes superpixels and bilateral filtering to create aesthetically pleasing, structure-aware pixel art.

## Example Output

| Original | K-Means | PIA |
| :---: | :---: | :---: |
| <img src="https://raw.githubusercontent.com/AlexandreBinninger/pixelization/refs/heads/main/assets/images/ferris_3d.png" alt="Original" width="256" height="256"> | <img src="https://github.com/AlexandreBinninger/pixelization/blob/main/assets/images/ferris_3d_Kmeans.png?raw=true" alt="KMeans" width="256" height="256" style="image-rendering: pixelated;"> | <img src="https://github.com/AlexandreBinninger/pixelization/blob/main/assets/images/ferris_3d_PIA.png?raw=true" alt="PIA" width="256" height="256" style="image-rendering: pixelated;"> |

## Installation

### As a CLI Tool

You can install the binary directly from crates.io (once published) or from source:

```bash
# From source
cargo install --path . --features cli
```

**⚠️ System Requirements (Read before installing):**
This library uses `ndarray-linalg`, which requires a BLAS backend.
* **Ubuntu/Debian:** `sudo apt update && sudo apt install pkg-config libssl-dev libopenblas-dev libx11-dev libxext-dev libxft-dev`
* **macOS:** `brew install openblas`
* **Windows:** You may need to install a pre-compiled LAPACK/BLAS binary or use `vcpkg`.

### As a Library

Add this to your `Cargo.toml`:

```toml
[dependencies]
pixelization = "0.1.1"
```


## Usage

### CLI

```bash
# Basic usage with PIA (default)
pixelize input.jpg --scale 4 --n_colors 8

# Using K-Means with specific output path
pixelize input.jpg -o output.png --method kmeans --n_colors 16 --scale 2
```

### Rust Code

```rust
use pixelization::{PIAPixelizer, Pixelizer};

fn main() {
    let img = image::open("assets/images/ferris_3d.png").unwrap();
    
    // Configure PIA
    let mut pia = PIAPixelizer::default();
    pia.set_verbose(true);

    // Pixelize: (image, width, height, number_of_colors)
    let result = pia.pixelize(&img, 64, 64, 8).unwrap();
    
    result.save("output.png").unwrap();
}
```

## Algorithms

* **K-Means:** Good for general color reduction. Fast.
* **PIA:** Excellent for preserving shapes and cartoons. It uses an iterative process involving superpixels, graph cuts, and palette optimization.

## License

This project is licensed under the MIT License.