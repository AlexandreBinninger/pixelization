extern crate image;
use image::DynamicImage;
use thiserror::Error;
use rand::Rng;

/// Defines how the image should be cropped when scaling.
pub enum CropMethod{
    /// No Crop
    NoCrop,
    /// Crop the center to match the target aspect ratio.
    CropEqual,
    /// Crop a random portion of the image.
    CropRandom
}

/// Helper function to scale an image to a specific size with cropping options.
/// Returns the new image and the effective dimensions.
pub fn scale_to_size<'a>(img: DynamicImage, scale: &u32, crop: CropMethod) -> (DynamicImage, (u32, u32)){
    let (w, h) = (img.width()/scale, img.height()/scale);
    match crop {
        CropMethod::NoCrop => (img, (w, h)),
        CropMethod::CropEqual => {
            let (new_w, new_h) = (w*scale, h*scale);
            let left = (img.width() - new_w) / 2;
            let top = (img.height() - new_h) / 2;
            (img.crop_imm(left, top, new_w, new_h), (w, h))
        },
        CropMethod::CropRandom => {
            let mut rng = rand::thread_rng();
            let (new_w, new_h) = (w*scale, h*scale);
            let left = rng.gen_range(0..(img.width() - new_w));
            let top = rng.gen_range(0..(img.height() - new_h));
            (img.crop_imm(left, top, new_w, new_h), (w, h))
        }
    }
}

/// The color space used for calculating distances between colors for K-means pixelization.
#[derive(Debug, Clone, Copy)]
pub enum ColorType {
    /// Standard RGB space.
    Rgb,
    /// CIELAB color space.
    Lab
}

/// Errors that can occur during the pixelization process.
#[derive(Debug, Error)]
pub enum PixelizationError {
    #[error("Dimension error: {0}")]
    DimensionError(String),
    #[error("Color error: {0}")]
    ColorError(String),
}

/// The main trait for any pixelization algorithm.
pub trait Pixelizer{
    /// Transforms the input image into a pixelated version.
    ///
    /// # Arguments
    /// * `img` - The source image.
    /// * `width` - Target width of the pixel grid.
    /// * `height` - Target height of the pixel grid.
    /// * `num_colors` - The maximum number of colors (palette size) to use.
    fn pixelize(&self, img: &DynamicImage, width : u32, height: u32,  num_colors : usize) -> Result<DynamicImage, PixelizationError>;
}

pub mod kmeans_pixelizer;
pub mod pia;