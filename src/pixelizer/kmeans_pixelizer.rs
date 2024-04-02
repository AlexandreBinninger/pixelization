use image::{DynamicImage, ImageBuffer};

use super::super::Pixelizer;
use kmeans_colors::{get_kmeans, Kmeans};
use palette::{rgb::Rgb, FromColor, IntoColor, Lab, Srgb};
use super::ColorType;

enum ColorVec {
    LabVec(Vec<Lab>),
    RgbVec(Vec<Rgb>),
}

pub struct KmeansPixelizer{
    num_runs: u32,
    max_iter : usize,
    color_type : ColorType
}

impl KmeansPixelizer {
    pub fn new(num_runs: u32, max_iter: usize, color_type: ColorType) -> Self {
        Self {
            num_runs,
            max_iter,
            color_type,
        }
    }
}


fn get_img_vec(rgb_image : ImageBuffer<image::Rgb<u8>, Vec<u8>>, color_type : &ColorType) -> ColorVec {
    match *color_type{
        ColorType::Lab => {
            ColorVec::LabVec(rgb_image.enumerate_pixels().map(|(_, _, pixel)| {
                let rgb_color = Srgb::new(pixel[0] as f32 / 255.0, pixel[1] as f32 / 255.0, pixel[2] as f32 / 255.0)
                    .into_linear();        
                let lab_color: Lab = rgb_color.into_color();
        
                lab_color
            }).collect::<Vec<Lab>>())
        },
        ColorType::Rgb =>{
            ColorVec::RgbVec(rgb_image.enumerate_pixels().map(|(_, _, pixel)| {
                let rgb_color = Srgb::new(pixel[0] as f32 / 255.0, pixel[1] as f32 / 255.0, pixel[2] as f32 / 255.0)
                    .into_linear();                
                let srgb_color : Rgb = rgb_color.into_color();
                srgb_color
            }).collect::<Vec<Rgb>>())
        }
    }
}

fn kmeans_lab(lab_vec: Vec<Lab>, num_colors: &usize, num_runs: &u32, max_iter: &usize) -> (Vec<Srgb<u8>>, Vec<u8>){
    let mut result = Kmeans::<Lab>::new();
    for _ in 0..*num_runs{
        let run_result = get_kmeans(
            *num_colors,
            *max_iter,
            5.0,
            false,
            &lab_vec,
            42
        );
        if run_result.score < result.score {
            result = run_result;
        }
    }
    let rgb_out = result.centroids
    .iter()
    .map(|x| Srgb::from_color(*x).into_format())
    .collect::<Vec<Srgb<u8>>>();
    let indices = result.indices;
    (rgb_out, indices)
}

fn kmeans_rgb(rgb_vec: Vec<Rgb>, num_colors: &usize, num_runs: &u32, max_iter: &usize) -> (Vec<Srgb<u8>>, Vec<u8>){
    let mut result = Kmeans::<Rgb>::new();
    for _ in 0..*num_runs{
        let run_result = get_kmeans(
            *num_colors,
            *max_iter,
            5.0,
            false,
            &rgb_vec,
            42
        );
        if run_result.score < result.score {
            result = run_result;
        }
    }
    let rgb_out = result.centroids
    .iter()
    .map(|x| Srgb::from_color(*x).into_format())
    .collect::<Vec<Srgb<u8>>>();
    let indices = result.indices;
    (rgb_out, indices)
}

impl Pixelizer for KmeansPixelizer{
    fn pixelize(&self, img: &DynamicImage,  width : &u32, height: &u32,  num_colors : &usize) -> DynamicImage{
        let img_width = img.width();
        let img_height =  img.height();
        assert!(*width <= img_width && *height <= img_height, "Provided width and height must be smaller or equal to the image dimensions");

        let rgb_img: ImageBuffer<image::Rgb<u8>, Vec<u8>> = img.to_rgb8();

        let img_vec = get_img_vec(rgb_img, &self.color_type);


        let (colors_palette, indices_color) = match img_vec{
            ColorVec::LabVec(lab_vec) =>{
                kmeans_lab(lab_vec, &num_colors, &self.num_runs, &self.max_iter)
            },
            ColorVec::RgbVec(rgb_vec) =>{
                kmeans_rgb(rgb_vec, &num_colors, &self.num_runs, &self.max_iter)
            }
        };

        let mut img_buffer = ImageBuffer::new(*width, *height);
        for (x, y, pixel) in img_buffer.enumerate_pixels_mut(){
            // x in 0..width, y in 0..height
            let start_x_orig = (img_width/ *width) * x;
            let end_x_orig = (img_width/ *width) * (x+1);
            let start_y_orig = (img_height/ *height) * y;
            let end_y_orig = (img_height/ *height) * (y+1);


            let mut color_indices = vec![0 as usize; *num_colors];
            for x_orig in start_x_orig..end_x_orig{
                for y_orig in start_y_orig..end_y_orig{
                    let index = (y_orig*img_width + x_orig) as usize;
                    let color_idx = indices_color[index] as usize;
                    color_indices[color_idx] += 1;
                }
            }
            
            let index_max = match color_indices.iter().enumerate().max_by_key(|(_, item)| *item) {
                Some((index, _)) => index,
                None => 0
            };
            
            if let Some(color) = colors_palette.get(index_max){
                *pixel = image::Rgb([color.red, color.green, color.blue])
            }
        }
        let dynamic_image = DynamicImage::ImageRgb8(img_buffer);
        dynamic_image   
    }
}


#[cfg(test)]
mod tests {
    use image::{DynamicImage, ImageError};

    use crate::{pixelizer::kmeans_pixelizer::ColorType, Pixelizer};

    #[test]
    fn test_size() {
        let img = image::open("examples/images/ferris_3d.png").unwrap();
        let pixelizer = super::KmeansPixelizer{
            num_runs: 4,
            max_iter: 20,
            color_type: ColorType::Lab
        };
        let width = 64;
        let height = 64;
        let num_colors = 8;
        let pixelized = pixelizer.pixelize(&img, &width, &height, &num_colors);

        let path_out = "examples/images/ferris_3d_pixelized.png";
        pixelized.save(path_out).unwrap();

        let size_pixelized = (pixelized.width(), pixelized.height());
        assert_eq!(size_pixelized, (width, height));
    }

    fn equal_dynamic_image(img1: &DynamicImage, img2: &DynamicImage) -> Result<bool, ImageError>{
        if img1.width() != img2.width() || img1.height() != img2.height(){
            return Ok(false);
        }
        let img_buf_1 = img1.to_rgb8();
        let img_buf_2 = img2.to_rgb8();
        Ok(img_buf_1.as_raw() == img_buf_2.as_raw())
    }

    #[test]
    fn test_uniform(){
        // Uniform image should be pixelized exactly the same, if given the same width and height
        let img = image::open("examples/images/uniform.png").unwrap();
        let pixelizer = super::KmeansPixelizer{
            num_runs: 8,
            max_iter: 20,
            color_type: ColorType::Rgb
        };
        let width = img.width();
        let height = img.height();
        let num_colors = 8;
        let pixelized = pixelizer.pixelize(&img, &width, &height, &num_colors);

        let path_out = "examples/images/uniform_pixelized.png";
        pixelized.save(path_out).unwrap();

        let size_pixelized = (pixelized.width(), pixelized.height());
        assert_eq!(size_pixelized, (width, height));
        if let Ok(b) = equal_dynamic_image(&img, &pixelized){
            assert!(b, "the uniform image is not pixelized to itself");
        }
    }
}