mod pixelizer;
pub use pixelizer::Pixelizer;
pub use pixelizer::kmeans_pixelizer::{KmeansPixelizer, ColorType};
pub use pixelizer::{scale_to_size, CropMethod};



#[cfg(test)]
mod tests {
    #[test]
    fn read_image() {
        let img = image::open("examples/images/ferris_3d.png").unwrap();
        assert_eq!(img, img);
    }
}
