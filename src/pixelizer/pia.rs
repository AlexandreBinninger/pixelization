// Implementation of Pixelated Image Abstraction (PIA)
// Link: https://pixl.cs.princeton.edu/pubs/Gerstner_2012_PIA/index.php

use image::{DynamicImage};

use super::super::Pixelizer;
use palette::{rgb::Rgb, FromColor, IntoColor, Lab, Srgb};
use core::num;
use std::collections::HashSet;

pub struct PIAPixelizer{
    temperature_init: Option<f32>,
    temperature_final: f32, // default should be 1.0
    m : f32, // coefficient for the cost function, eq (1) of the paper,
    alpha: f32, // efault should be 0.7
    epsilon_palette: f32, // default is 1
    epsilon_cluster: f32, // default is 0.25
    post_process_saturation: f32, // default is 1.1 in the paper
}

impl PIAPixelizer {
    // m is set to 45 in the paper
    // post_process_saturation is set to 1.1 in the paper
    pub fn new(temperature_init: Option<f32>, temperature_final: f32, m: f32, alpha: f32, epsilon_palette: f32, epsilon_cluster: f32, post_process_saturation: f32) -> Self {
        Self {
            temperature_init,
            temperature_final,
            m,
            alpha,
            epsilon_palette,
            epsilon_cluster,
            post_process_saturation
        }
    }
}

#[derive(Debug)]
struct SuperPixel{
    position: (f32, f32),
    palette_color: Lab,
    pixels: HashSet<(u32, u32)>,
    original_position: (f32, f32),
    p_s: f32,
    p_c: Vec<f32>,
    sp_color: Lab
}

fn lab_distance(lab1: Lab, lab2: Lab) -> f32{
    let l_diff = lab1.l - lab2.l;
    let a_diff = lab1.a - lab2.a;
    let b_diff = lab1.b - lab2.b;

    (l_diff * l_diff + a_diff * a_diff + b_diff * b_diff).sqrt()
}

impl SuperPixel{
    fn new(position: (f32, f32), palette_color: Lab, num_sp: usize) -> Self{
        let pixels = HashSet::new();
        let original_position = position;
        let p_s = 1.0 / num_sp as f32;
        let sp_color = Lab::default();

        let p_c = vec![0.5; 2];

        Self{
            position,
            palette_color,
            pixels,
            original_position,
            p_s,
            p_c,
            sp_color
        }
    }

    // eq (1) of the paper
    fn cost(&self, x: u32, y: u32, img_lab : Vec<Vec<Lab>>, m: f32, nb_pixels_in : u32, nb_pixels_out : u32) -> f32{
        let lab_color = img_lab[y as usize][x as usize];
        let d_lab = lab_distance(self.palette_color, lab_color);
        let d_spatial = ((self.position.0 - x as f32).powi(2) + (self.position.1 - y as f32).powi(2)).sqrt();

        d_lab + m * ((nb_pixels_out as f32) / (nb_pixels_in as f32)).sqrt() * d_spatial
    }

    fn add_pixel(&mut self, x: u32, y: u32){
        self.pixels.insert((x, y));
    }

    fn clear_pixels(&mut self){
        self.pixels.clear();
    }

    fn update_pos(&mut self){
        assert!(self.pixels.len() > 0);
        self.position = self.pixels.iter().fold((0.0, 0.0), |acc, (x, y)| {
            (acc.0 + *x as f32, acc.1 + *y as f32)
        });
        self.position = (self.position.0 / self.pixels.len() as f32, self.position.1 / self.pixels.len() as f32);
    }

    fn update_sp_color(&mut self, img: &Vec<Vec<Lab>>){
        assert!(self.pixels.len() > 0);
        self.sp_color = self.pixels.iter().fold(Lab::default(), |acc, (x, y)| {
            acc+img[*y as usize][*x as usize]
        }) / self.pixels.len() as f32;
    }
}


#[derive(Debug)]
struct Palette{
    colors: Vec<Lab>,
    probabilities: Vec<f32> // invariants: the sum of the probabilities is 1
}

impl Palette{
    fn new(init_color : Lab) -> Self{
        let colors = vec![init_color; 2];
        let probabilities = vec![0.5; 2];
        Self{
            colors,
            probabilities
        }
    }
}

struct PIAGlobal{
    superpixels: Vec<Vec<SuperPixel>>,
    clusters: Vec<(usize, usize)>,
    palette: Palette,
}

impl PIAGlobal{
    fn new(w: u32, h: u32, lab_img: &Vec<Vec<Lab>>) -> Self{
        let num_sp = (w * h) as usize;
        let mut superpixels = Vec::with_capacity(num_sp);

        let width_orig = lab_img[0].len();
        let height_orig = lab_img.len();

        let width_ratio = width_orig as f32 / (w+1) as f32;
        let height_ratio = height_orig as f32 / (h+1) as f32;

        // get average color
        let sum_color = lab_img.iter().map(|row| {
            row.iter().fold((0.0, 0.0, 0.0), |acc, color| {
                (acc.0 + color.l, acc.1 + color.a, acc.2 + color.b)
            })
            }).fold((0.0, 0.0, 0.0), |acc, color| {
                (acc.0 + color.0, acc.1 + color.1, acc.2 + color.2)
            });
        let total_pixels = (width_orig * height_orig) as f32;
        let avg_lab_color = Lab::new(sum_color.0 / total_pixels, sum_color.1 / total_pixels, sum_color.2 / total_pixels);

        for y in 1..(h+1){
            let mut row_superpixels = Vec::with_capacity(w as usize);
            for x in 1..(w+1){
                let position = (x as f32 * width_ratio, y as f32 * height_ratio);
                row_superpixels.push(SuperPixel::new(position, avg_lab_color, num_sp));
            }
            superpixels.push(row_superpixels);
        }
        let clusters = vec![(0, 1); 1];
        let palette = Palette::new(avg_lab_color);
        Self{
            superpixels,
            clusters,
            palette
        }
    }

    fn refine_superpixels(&mut self, img: &Vec<Vec<Lab>>, m: f32){
        
    }

    fn associate(&mut self, img: &Vec<Vec<Lab>>){
        
    }

    fn refine_palette(&mut self, img: &Vec<Vec<Lab>>, epsilon_palette: &f32) -> bool{
        true
    }

    fn expand_palette(&mut self, img: &Vec<Vec<Lab>>, num_colors: &usize, epsilon_cluster: &f32){    
    }

    fn get_img(&self, saturation: &f32) -> DynamicImage{
        DynamicImage::new_rgb8(1, 1)
    }

    fn get_current_nb_colors(&self) -> usize{
        self.clusters.len()
    }
}


impl Pixelizer for PIAPixelizer{
    fn pixelize(&self, img: &DynamicImage, width : &u32, height: &u32,  num_colors : &usize) -> DynamicImage{
        let rgb_image = img.to_rgb8();
        let width_orig = rgb_image.width();
        let height_orig = rgb_image.height();
        let lab_vec = rgb_image.enumerate_pixels().map(|(x, y, pixel)| {
            let rgb_color = Srgb::new(pixel[0] as f32 / 255.0, pixel[1] as f32 / 255.0, pixel[2] as f32 / 255.0)
                .into_linear();        
            let lab_color: Lab = rgb_color.into_color();
    
            (x, y, lab_color)
        }).fold(vec![vec![Lab::default(); width_orig as usize]; height_orig as usize], | mut acc, (x, y, pixel)| {
            acc[y as usize][x as usize] = pixel;
            return acc;
        });

        let mut global = PIAGlobal::new(*width, *height, &lab_vec);

        let mut temperature = match self.temperature_init {
            Some(t) => t,
            None => {
                35.0 //TODO: PCA here
            }
        };

        while temperature > self.temperature_final{            
            // refine superpixels
            global.refine_superpixels(&lab_vec, self.m);
            // associate
            global.associate(&lab_vec);
            // refine colors in the palette
            let to_expand = global.refine_palette(&lab_vec, &self.epsilon_palette);

            // if palette converged, reduce temperature and expand palette if necessary
            if to_expand{
                temperature *= self.alpha;
                if global.get_current_nb_colors() < *num_colors{
                    global.expand_palette(&lab_vec, num_colors, &self.epsilon_cluster);
                }
            }
        }
        // post process
        let img_out = global.get_img(&self.post_process_saturation);

        // return image
        img_out
    }
}


#[cfg(test)]
mod tests {
    use image::{DynamicImage, ImageError};

    use crate::pixelizer::pia::PIAPixelizer;
    use super::Pixelizer;

    #[test]
    fn test_size() {
        let img = image::open("examples/images/ferris_3d.png").unwrap();
        let pixelizer = PIAPixelizer{
            temperature_init: Some(35.0),
            temperature_final: 1.0,
            m: 45.0,
            alpha: 0.7,
            epsilon_palette: 1.0,
            epsilon_cluster: 0.25,
            post_process_saturation: 1.1
        };
        let width = 8;
        let height = 8;
        let num_colors = 8;
        let pixelized = pixelizer.pixelize(&img, &width, &height, &num_colors);

        let path_out = "examples/images/ferris_3d_PIA.png";
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

    // #[test]
    // fn test_uniform(){
    //     // Uniform image should be pixelized exactly the same, if given the same width and height
    //     let img = image::open("examples/images/uniform.png").unwrap();
    //     let pixelizer = super::KmeansPixelizer{
    //         num_runs: 8,
    //         max_iter: 20,
    //         color_type: ColorType::Rgb
    //     };
    //     let width = img.width();
    //     let height = img.height();
    //     let num_colors = 8;
    //     let pixelized = pixelizer.pixelize(&img, &width, &height, &num_colors);

    //     let path_out = "examples/images/uniform_pixelized.png";
    //     pixelized.save(path_out).unwrap();

    //     let size_pixelized = (pixelized.width(), pixelized.height());
    //     assert_eq!(size_pixelized, (width, height));
    //     if let Ok(b) = equal_dynamic_image(&img, &pixelized){
    //         assert!(b, "the uniform image is not pixelized to itself");
    //     }
    // }
}