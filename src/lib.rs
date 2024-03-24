mod pixelizer;
pub use pixelizer::Pixelizer;
pub use pixelizer::kmeans_pixelizer::KmeansPixelizer;



#[cfg(test)]
mod tests {
    #[test]
    fn read_image() {
        let img = image::open("examples/images/ferris_3d.png").unwrap();
        assert_eq!(img, img);
    }
}
