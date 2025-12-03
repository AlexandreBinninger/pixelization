//! # Pixelization
//!
//! `pixelization` is a high-performance image quantization and pixelization library.
//! It implements standard strategies like **K-Means Clustering** and complex, structure-aware
//! strategies like **Pixelated Image Abstraction (PIA)**.
//!
//! ## Example
//!
//! ```no_run
//! use pixelization::{KmeansPixelizer, Pixelizer, ColorType};
//!
//! let img = image::open("input.png").unwrap();
//! let pixelizer = KmeansPixelizer::new(3, 20, ColorType::Lab);
//! 
//! // Downscale to 64x64 and reduce to 8 colors
//! let result = pixelizer.pixelize(&img, 64, 64, 8).unwrap();
//! result.save("output.png").unwrap();
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
