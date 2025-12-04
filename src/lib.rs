//! # Pixelization
//!
//! `pixelization` is an image quantization and pixelization library.
//! It implements standard strategies like **K-Means Clustering** and complex, structure-aware
//! strategies like **Pixelated Image Abstraction (PIA)**.
//!
//! ## Examples
//!
//! ### Basic K-Means
//! Fast and simple color reduction.
//!
//! ```no_run
//! use pixelization::{KmeansPixelizer, Pixelizer, ColorType};
//!
//! let img = image::open("input.png").unwrap();
//! let pixelizer = KmeansPixelizer::new(3, 20, ColorType::Lab);
//! 
//! // Downscale to 64x64 and reduce to 8 colors
//! let result = pixelizer.pixelize(&img, 64, 64, 8).unwrap();
//! result.save("output_kmeans.png").unwrap();
//! ```
//! 
//! ### Pixelated Image Abstraction (PIA)
//! Structure-aware pixelization using the default parameters from the original paper that you can find at <https://pixl.cs.princeton.edu/pubs/Gerstner_2012_PIA/index.php>.
//! 
//! ```no_run
//! use pixelization::{PIAPixelizer, Pixelizer};
//! 
//! let img = image::open("input.png").unwrap();
//! 
//! // Use default parameters (m=45.0, alpha=0.7, etc.)
//! // This is recommended for most images.
//! let mut pixelizer = PIAPixelizer::default();
//! 
//! // Optional: Enable verbose logging to see iteration progress
//! pixelizer.set_verbose(true);
//! 
//! // PIA is computationally intensive; small target sizes (e.g., 64x64) are recommended.
//! let result = pixelizer.pixelize(&img, 64, 64, 8).unwrap();
//! result.save("output_pia.png").unwrap();
//! ```


mod pixelizer;
// Re-export specific items to make them easy to access
pub use pixelizer::{Pixelizer, ColorType, CropMethod, scale_to_size, PixelizationError};
pub use pixelizer::kmeans_pixelizer::KmeansPixelizer;
pub use pixelizer::pia::PIAPixelizer;


#[cfg(test)]
mod tests {
    #[test]
    fn read_image() {
        let img = image::open("assets/images/ferris_3d.png").unwrap();
        assert_eq!(img, img);
    }
}
