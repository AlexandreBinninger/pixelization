// Implementation of Pixelated Image Abstraction (PIA)
// Link: https://pixl.cs.princeton.edu/pubs/Gerstner_2012_PIA/index.php

use image::{DynamicImage, Rgba, GenericImage};
use ndarray::{Array1, Array2};
use ndarray_linalg::{Eigh, UPLO};

use super::super::Pixelizer;
use palette::{FromColor, IntoColor, Lab, Srgb};
use std::collections::HashSet;
use rayon::prelude::*;
use std::sync::Mutex;



pub struct PIAPixelizer{
    temperature_init: Option<f32>,
    temperature_final: f32, // default should be 1.0
    m : f32, // coefficient for the cost function, eq (1) of the paper,
    alpha: f32, // default should be 0.7
    epsilon_palette: f32, // default is 1
    epsilon_cluster: f32, // default is 0.25
    post_process_saturation: f32, // default is 1.1 in the paper
    laplacian_smoothing_factor: f32,
    bilateral_filter_sigma_spatial: f32,
    bilateral_filter_sigma_range: f32,
    color_perturbation_coefficient: f32,
    verbose: bool
}

impl Default for PIAPixelizer {
    fn default() -> Self {
        Self{
            temperature_init: None,
            temperature_final: 1.0,
            m: 45.0, // m is set to 45 in the paper
            alpha: 0.7,
            epsilon_palette: 1.0,
            epsilon_cluster: 0.25,
            post_process_saturation: 1.1, // post_process_saturation is set to 1.1 in the paper
            laplacian_smoothing_factor: 0.4,
            bilateral_filter_sigma_spatial: 0.87,
            bilateral_filter_sigma_range: 0.87,
            color_perturbation_coefficient: 0.8,
            verbose: false
        }
    }
}

impl PIAPixelizer {
    pub fn new(temperature_init: Option<f32>, temperature_final: f32, m: f32, alpha: f32, verbose:bool) -> Self {
        Self {
            temperature_init,
            temperature_final,
            m,
            alpha,
            verbose,
            ..Default::default()
        }
    }

    pub fn set_verbose(&mut self, verbose: bool){
        self.verbose = verbose;
    }
}

#[derive(Debug)]
struct SuperPixel{
    position: (f32, f32),
    palette_color: Lab,
    pixels: HashSet<(u32, u32)>,
    p_s: f32,
    p_c: Vec<f32>,
    sp_color: Lab
}

fn lab_distance(lab1: &Lab, lab2: &Lab) -> f32{
    let l_diff = lab1.l - lab2.l;
    let a_diff = lab1.a - lab2.a;
    let b_diff = lab1.b - lab2.b;
    (l_diff * l_diff + a_diff * a_diff + b_diff * b_diff).sqrt()
}

impl SuperPixel{
    fn new(position: (f32, f32), palette_color: Lab, num_sp: usize) -> Self{
        let pixels = HashSet::new();
        let p_s = 1.0 / num_sp as f32;
        let sp_color = Lab::default();

        let p_c = vec![0.5; 2];

        Self{
            position,
            palette_color,
            pixels,
            p_s,
            p_c,
            sp_color
        }
    }

    // eq (1) of the paper
    fn cost(&self, x: u32, y: u32, img_lab : &Vec<Vec<Lab>>, m: f32, nb_pixels_in : u32, nb_pixels_out : u32) -> f32{
        let lab_color = img_lab[y as usize][x as usize];
        let d_lab = lab_distance(&self.palette_color, &lab_color);
        let d_spatial = ((self.position.0 - x as f32).powi(2) + (self.position.1 - y as f32).powi(2)).sqrt();

        d_lab + m * ((nb_pixels_out as f32) / (nb_pixels_in as f32)).sqrt() * d_spatial
    }

    fn add_pixels(&mut self, pixels: &Vec<(u32, u32)>){
        self.pixels.extend(pixels);
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

    fn normalize_probs(&mut self){
        let sum = self.p_c.iter().sum::<f32>();
        self.p_c = self.p_c.iter().map(|x| x / sum).collect();
    }

    fn update_palette_color(&mut self, colors: &Vec<Lab>){
        let highest_prob = self.p_c.iter().enumerate().max_by(|x, y| x.1.partial_cmp(y.1).unwrap()).unwrap().0;
        self.palette_color = colors[highest_prob];
    }
}


#[derive(Debug)]
struct Palette{
    colors: Vec<Lab>,
    probabilities: Vec<f32> // invariants: the sum of the probabilities should be 1
}

impl Palette{
    fn new(init_color : Lab, perturbation: Lab) -> Self{
        let mut colors = vec![init_color; 2];
        colors[1] += perturbation;
        let probabilities = vec![0.5; 2];
        Self{
            colors,
            probabilities
        }
    }

    fn conditional_prob(&self, color: Lab, temperature: f32) -> Vec<f32>{
        let mut probs = Vec::with_capacity(self.colors.len());
        for k in 0..self.colors.len(){
            let d_lab = lab_distance(&self.colors[k], &color);
            let prob = self.probabilities[k]*(-d_lab/temperature).exp();
            probs.push(prob);
        }
        probs
    }
}

fn bilateral_filter_lab(img_lab: &Vec<Vec<Lab>>, sigma_spatial: f32, sigma_range: f32) -> Vec<Vec<Lab>> {
    let height = img_lab.len();
    let width = img_lab.first().unwrap_or(&vec![]).len();
    let mut output = vec![vec![Lab::default(); width]; height];

    let filter_radius = (sigma_spatial * 3.0).ceil() as i32;

    for y in 0..height as i32 {
        for x in 0..width as i32 {
            let mut sum_weight = 0.0;
            let mut sum_l = 0.0;
            let mut sum_a = 0.0;
            let mut sum_b = 0.0;
            let center_color = img_lab[y as usize][x as usize];

            for ky in -filter_radius..=filter_radius {
                for kx in -filter_radius..=filter_radius {
                    let nx = x + kx;
                    let ny = y + ky;

                    if nx >= 0 && nx < width as i32 && ny >= 0 && ny < height as i32 {
                        let neighbor_color = img_lab[ny as usize][nx as usize];

                        let spatial_weight = (-((kx.pow(2) + ky.pow(2)) as f32) / (2.0 * sigma_spatial.powi(2))).exp();
                        let color_distance = lab_distance(&center_color, &neighbor_color);
                        let range_weight = (-(color_distance.powi(2)) / (2.0 * sigma_range.powi(2))).exp();

                        let weight = spatial_weight * range_weight;

                        sum_l += neighbor_color.l * weight;
                        sum_a += neighbor_color.a * weight;
                        sum_b += neighbor_color.b * weight;
                        sum_weight += weight;
                    }
                }
            }
            output[y as usize][x as usize] = Lab::new(sum_l / sum_weight, sum_a / sum_weight, sum_b / sum_weight);
        }
    }

    output
}


// Take a vector of Lab colors and return the eigenvector with the highest eigenvalue
fn pca_lab(lab_vec : &Vec<Lab>) -> (Array1<f64>, f64) {
    // Convert to ndarry
    let rows = lab_vec.len();
    let mut data : Array2<f64> = Array2::zeros((rows, 3));
    for (i, color) in lab_vec.iter().enumerate(){
        data[(i, 0)] = color.l as f64;
        data[(i, 1)] = color.a as f64;
        data[(i, 2)] = color.b as f64;
    }

    // Compute PCA: center, covariance matrix, eigen decomposition
    let means = data.mean_axis(ndarray::Axis(0)).unwrap();
    let centered_data = data - &means.insert_axis(ndarray::Axis(0));

    let covariance_matrix = centered_data.t().dot(&centered_data) / (rows as f64 - 1.0);
    let (eigenvalues, eigenvector) = covariance_matrix.eigh(UPLO::Upper).unwrap();

    (eigenvector.column(eigenvector.ncols() - 1).to_owned(), eigenvalues[eigenvalues.len() - 1])
}

struct PIAGlobal{
    superpixels: Vec<Vec<SuperPixel>>,
    clusters: Vec<(usize, usize)>,
    palette: Palette,
    critical_temperature : f32,
    perturbation: Lab
}

impl PIAGlobal{
    fn new(w: u32, h: u32, lab_img: &Vec<Vec<Lab>>, color_perturbation_coefficient: f32) -> Self{
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

        let img_lab_flatter : Vec<Lab> = lab_img.iter().flatten().cloned().collect();
        let (eigenvector, eigenvalue) = pca_lab(&img_lab_flatter);

        let clusters = vec![(0, 1); 1];
        let perturbation = Lab::from_components((eigenvector[0] as f32, eigenvector[1] as f32, eigenvector[2] as f32)) * color_perturbation_coefficient;
        let palette = Palette::new(avg_lab_color, perturbation);

        Self{
            superpixels,
            clusters,
            palette,
            critical_temperature: (2.0*eigenvalue as f32).sqrt(), // Section 4.1 in the paper. In the code, they take the square root 
            perturbation
        }
    }

    fn refine_superpixels(&mut self, img: &Vec<Vec<Lab>>, m: f32, laplacian_smoothing_factor: f32, bilateral_filter_sigma_spatial: f32, bilateral_filter_sigma_range: f32){
        for superpixel in self.superpixels.iter_mut().flatten(){
            superpixel.clear_pixels();
        }

        // Find best superpixel for each pixel
        let sp_pixels: Vec<_> = self.superpixels.iter_mut()
                                                .map(|row| row.iter_mut().map(|_| Vec::new()).collect::<Vec<_>>())
                                                .collect();
        let sp_pixels_mutex = Mutex::new(sp_pixels);

        img.par_iter().enumerate().for_each(|(y, row)| {
            row.iter().enumerate().for_each(|(x, _)| {
                let mut min_cost = std::f32::MAX;
                let mut min_sp_index = (0, 0);
                for (y_sp, superpixel_row) in self.superpixels.iter().enumerate(){
                    for (x_sp, superpixel) in superpixel_row.iter().enumerate(){
                        let cost = superpixel.cost(x as u32, y as u32, img, m, 1, 1);
                        if cost < min_cost{
                            min_cost = cost;
                            min_sp_index = (x_sp, y_sp);
                        }
                    }
                }
                let mut sp_pixels = sp_pixels_mutex.lock().unwrap();
                sp_pixels[min_sp_index.1][min_sp_index.0].push((x as u32, y as u32));
            });
        });

        let sp_pixels = sp_pixels_mutex.into_inner().unwrap();

        for (y, row) in sp_pixels.into_iter().enumerate(){
            for (x, pixels) in row.into_iter().enumerate(){
                self.superpixels[y][x].add_pixels(&pixels);
            }
        }

        // Refine colors and positions for superpixels
        for superpixel in self.superpixels.iter_mut().flatten(){
            superpixel.update_pos();
            superpixel.update_sp_color(img);
        }

        let in_image = |x: f32, y:f32| x >=0. && y >= 0. && x < img[0].len() as f32 && y < img.len() as f32;

        // Laplacian smoothing
        let x_shift = vec![-1., 0., 1., 0.];
        let y_shift = vec![0., -1., 0., 1.];

        for row in self.superpixels.iter_mut(){
            for superpixel in row.iter_mut(){
                let (mut new_x, mut new_y, mut n) = (0.0, 0.0, 0);
                for (dx, dy) in x_shift.iter().zip(y_shift.iter()){
                    let x_neigh = superpixel.position.0 + dx;
                    let y_neigh = superpixel.position.1 + dy;
                    if in_image(x_neigh, y_neigh){
                        new_x += x_neigh;
                        new_y += y_neigh;
                        n += 1;
                    }
                }
                new_x /= n as f32;
                new_y /= n as f32;
                superpixel.position = (new_x * laplacian_smoothing_factor + (1.0-laplacian_smoothing_factor) * superpixel.position.0, new_y * laplacian_smoothing_factor + (1.0-laplacian_smoothing_factor) * superpixel.position.1);
            }
        }

        // Bilateral filtering
        let img_from_superpixels : Vec<Vec<Lab>> = self.superpixels.iter().map(|row|{
            row.iter().map(|superpixel| superpixel.sp_color).collect()
        }).collect();

        let img_filtered = bilateral_filter_lab(&img_from_superpixels, bilateral_filter_sigma_spatial, bilateral_filter_sigma_range);

        self.superpixels.iter_mut().enumerate().for_each(|(y, row)| {
            row.iter_mut().enumerate().for_each(|(x, superpixel)| {
                superpixel.sp_color = img_filtered[y][x];
            });
        });
    }

    fn associate(&mut self, temperature: f32){
        self.superpixels.iter_mut().flatten().for_each(|superpixel| {
            let probs = self.palette.conditional_prob(superpixel.sp_color, temperature);
            superpixel.p_c = probs;
            superpixel.normalize_probs();
            superpixel.update_palette_color(&self.palette.colors);
        });

        self.palette.probabilities = vec![0.0; self.palette.probabilities.len()];
        for superpixel in self.superpixels.iter().flatten(){
            for (k, prob) in superpixel.p_c.iter().enumerate(){
                self.palette.probabilities[k] += prob * superpixel.p_s;
            }
        }
    }

    fn refine_palette(&mut self, epsilon_palette: &f32) -> bool{
        let mut total_change: f32 = 0.0;
        let mut updated_colors = Vec::with_capacity(self.palette.colors.len());
        for (k, color) in self.palette.colors.iter_mut().enumerate(){
            let palette_probability = self.palette.probabilities[k];
            let acc = self.superpixels.iter().flatten().fold((0., 0., 0.), |acc, superpixel| {
                let prob = superpixel.p_c[k] * superpixel.p_s / palette_probability;
                (acc.0 + superpixel.sp_color.l * prob, acc.1 + superpixel.sp_color.a * prob, acc.2 + superpixel.sp_color.b * prob)
            });
            let new_color = Lab::new(acc.0, acc.1, acc.2);
            let change = lab_distance(&new_color, color);
            updated_colors.push(new_color);
            total_change += change;
        }

        self.palette.colors = updated_colors;
        return total_change < *epsilon_palette; // It seems like it is not necessary to divide by the number of colors, according to original source code.
    }

    fn expand_palette(&mut self,num_colors: &usize, epsilon_cluster: &f32){
        let cluster_size = self.clusters.len();
        if cluster_size < *num_colors{
            for k in 0..cluster_size{
                if self.clusters.len() < *num_colors{
                    let (i, j) = self.clusters[k];
                    let color_i = self.palette.colors[i];
                    let color_j = self.palette.colors[j];
                    let prob_i = self.palette.probabilities[i]/2.0;
                    let prob_j = self.palette.probabilities[j]/2.0;
                    if lab_distance(&color_i, &color_j) > *epsilon_cluster{
                        self.palette.colors.push(color_i);
                        self.palette.colors.push(color_j);
                        self.palette.probabilities.push(prob_i);
                        self.palette.probabilities.push(prob_j);
                        self.palette.probabilities[i] = prob_i;
                        self.palette.probabilities[j] = prob_j;
                        self.clusters.push((i, self.palette.colors.len() - 2));
                        self.clusters[k] = (j, self.palette.colors.len() - 1);
                    }
                }
            }
        }
        let cluster_size = self.clusters.len();
        if cluster_size >= *num_colors{
            let mut new_palette_colors = Vec::with_capacity(*num_colors);
            let mut new_palette_probs = Vec::with_capacity(*num_colors);
            for k in 0..cluster_size{
                let (i, j) = self.clusters[k];
                let color_i = self.palette.colors[i];
                let color_j = self.palette.colors[j];
                let prob_i = self.palette.probabilities[i];
                let prob_j = self.palette.probabilities[j];
                let new_color = Lab::new((color_i.l + color_j.l) / 2.0, (color_i.a + color_j.a) / 2.0, (color_i.b + color_j.b) / 2.0);
                new_palette_colors.push(new_color);
                new_palette_probs.push(prob_i + prob_j);
            }
            self.palette.colors = new_palette_colors;
            self.palette.probabilities = new_palette_probs;
        } else{
            for k in 0..cluster_size{
                let current_color = self.palette.colors[self.clusters[k].1];
                let perturbed_color =current_color + self.perturbation;
                self.palette.colors[self.clusters[k].1] = perturbed_color;
            }
        }
    }

    fn get_img(&self, saturation: &f32) -> DynamicImage{
        let width = self.superpixels[0].len();
        let height = self.superpixels.len();
        let mut img = DynamicImage::new_rgb8(width as u32, height as u32);
        for (y, row) in self.superpixels.iter().enumerate(){
            for (x, superpixel) in row.iter().enumerate(){
                let (l, mut a, mut b) = superpixel.palette_color.into_components();
                a *= *saturation;
                b *= *saturation;
                let color = Lab::new(l, a, b);
                let srgb_color: Srgb<u8> = Srgb::from_color(color).into_format();
                let rgba_color = Rgba([srgb_color.red, srgb_color.green, srgb_color.blue, 255]);
                img.put_pixel(x as u32, y as u32, rgba_color);
            }
        }
        img
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

        let mut global = PIAGlobal::new(*width, *height, &lab_vec, self.color_perturbation_coefficient);

        let mut temperature = match self.temperature_init {
            Some(t) => t,
            None => {
                1.1 * global.critical_temperature // Section 4.1 in the paper
            }
        };

        let mut iter = 0;
        while temperature > self.temperature_final{
            if self.verbose{
                println!("Iteration: {}", iter);
                println!("Temperature: {}", temperature);
            }
            iter += 1;
            // refine superpixels
            global.refine_superpixels(&lab_vec, self.m, self.laplacian_smoothing_factor, self.bilateral_filter_sigma_spatial, self.bilateral_filter_sigma_range);
            // associate
            global.associate(temperature);
            // refine colors in the palette
            let to_expand = global.refine_palette(&self.epsilon_palette);

            // if palette converged, reduce temperature and expand palette if necessary
            if to_expand{
                if self.verbose{
                    println!("Expanding palette");
                }
                temperature *= self.alpha;
                if global.get_current_nb_colors() < *num_colors{
                    global.expand_palette(num_colors, &self.epsilon_cluster);
                }
            }
            else {
                if self.verbose{
                    println!("Not expanding palette");
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
            temperature_init: None,
            temperature_final: 1.0,
            m: 45.0,
            alpha: 0.7,
            epsilon_palette: 1.0,
            epsilon_cluster: 0.25,
            post_process_saturation: 1.1,
            laplacian_smoothing_factor: 0.4,
            bilateral_filter_sigma_spatial: 0.87,
            bilateral_filter_sigma_range: 0.87,
            color_perturbation_coefficient: 0.8,
            verbose: true
        };
        let width = 64;
        let height = 64;
        let num_colors = 8;
        let pixelized = pixelizer.pixelize(&img, &width, &height, &num_colors);

        let path_out = "examples/images/ferris_3d_PIA_64_8.png";
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
        let pixelizer = PIAPixelizer{
            temperature_init: Some(35.0),
            temperature_final: 1.0,
            m: 45.0,
            alpha: 0.7,
            epsilon_palette: 1.0,
            epsilon_cluster: 0.25,
            post_process_saturation: 1.0,
            laplacian_smoothing_factor: 0.4,
            bilateral_filter_sigma_spatial: 0.87,
            bilateral_filter_sigma_range: 0.87,
            color_perturbation_coefficient: 0.8,
            verbose: true
        };
        let width = img.width()/2; // can't have the same size
        let height = img.height()/2; // can't have the same size
        let num_colors = 8;
        let pixelized = pixelizer.pixelize(&img, &width, &height, &num_colors);
        let path_out = "examples/images/uniform_pixelized_PIA.png";
        pixelized.save(path_out).unwrap();
        let pixelized = pixelized.resize(width*2, height*2, image::imageops::FilterType::Nearest);

        let size_pixelized = (pixelized.width(), pixelized.height());
        assert_eq!(size_pixelized, (width*2, height*2));
        if let Ok(b) = equal_dynamic_image(&img, &pixelized){
            assert!(b, "the uniform image is not pixelized to itself");
        }
    }
}