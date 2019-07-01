use rand::prelude::*;
use std::cmp::Ordering;
use rayon::prelude::*;
const LEARNING_RATE: f64 = 0.5;

struct Node {
    weights: Vec<f64>,
    x: i32,
    y: i32
}

impl Node {
    fn new(dimensions: usize, x: i32, y: i32) -> Self {
        let mut rng = rand::thread_rng();
        let mut weights = vec![0.0 as f64; dimensions];
        for i in 0 .. dimensions {
            weights[i] = rng.gen();
        }
        Node {
            weights: weights,
            x: x, y: y
        }
    }

    fn get_weights(&self) -> &[f64] {
        &self.weights
    }

    fn get_weight_distance(&self, inputs: &[f64]) -> f64 {
        if self.get_weights().len() != inputs.len() {
            panic!("Data and dimensions should be the same size")
        }

        let mut distance: f64 = 0.0;
        let self_weights = self.get_weights();

        for i in 0 .. self_weights.len() {
            distance += (inputs[i] - self_weights[i]).powi(2);
        }
       
        distance
    }

    fn get_euclid_distance_squared(&self, x: i32, y: i32) -> f64 {
        ((x - self.x) as f64).powi(2) + 
            ((y - self.y) as f64).powi(2)
    }

    fn adjust_weights(&mut self, data: &[f64], learning_rate: f64, influence: f64) { 
     
        for (value, target) in self.weights.iter_mut().zip(data) {
            *value += learning_rate * influence * (*target - *value)
        }

    }
}


struct SelfOrganizingMap {
    iterations: u32,
    width: usize,
    height: usize,
    nodes: Vec<Node>
}
impl SelfOrganizingMap {

    fn new(dimensions: usize, iterations: u32, width: usize, height: usize) -> SelfOrganizingMap {
        let mut nodes = Vec::<Node>::with_capacity(width * height);
        for i in 0 .. width {
            for j in 0 .. height {
                nodes.push(Node::new(dimensions, i as i32, j as i32));
            }
        }
        SelfOrganizingMap { 
            iterations: iterations,
            width: width,
            height: height,
            nodes: nodes 
        }
    }

    fn find_best_match_position(&self, data: &[f64]) -> (i32, i32) {
        
        let (node, _dist) = self.nodes.par_iter()
            .map(|node| {
                let distance = node.get_weight_distance(data);
                (node, distance)
            })
           .min_by(|(_, dist_a), (_, dist_b)| {
                match dist_a.partial_cmp(dist_b) {
                    Some(x) => x,
                    None => Ordering::Greater
                }
            }).unwrap();

        (node.x, node.y)
    }

    fn epoch<T>(&mut self, data: &[T], iteration: u32) 
        where T: AsRef<[f64]> {

        let iteration = iteration as f64;
        let total_iterations = self.iterations as f64;
        let map_radius = (self.width as f64).max(self.height as f64) / 2.0;
        let time_constant = total_iterations / map_radius.ln();
        let learning_rate = LEARNING_RATE * (-iteration / total_iterations).exp();
       
        let randomly_chosen : &[f64] = 
            data[rand::thread_rng().gen_range(0, data.len())].as_ref();

        let (bmu_x, bmu_y) = self.find_best_match_position(randomly_chosen);

        let neighborhood_radius = map_radius * (-iteration / time_constant).exp();
        
        self.nodes.par_iter_mut().for_each(|node| {
            let dist_to_bmu = node.get_euclid_distance_squared(bmu_x, bmu_y);

            let radius = neighborhood_radius.powi(2);

            if dist_to_bmu < radius {
                let influence = ((-dist_to_bmu) / (2.0 * radius)).exp();
                node.adjust_weights(randomly_chosen, learning_rate, influence);
            }
        })

    }

    fn train<T, F>(&mut self, data: &[T], mut visitor: F)
        where T: AsRef<[f64]>, F: FnMut(&[Node], u32) -> ()
    {
        for i in 0 .. self.iterations {
            self.epoch(data, i);
            visitor(&self.nodes, i);
        }
    }
}


fn normalize(mut color: Vec<f64>, max: f64) -> Vec<f64> {

    let mult = 1.0 / max;
    for col in color.iter_mut() {
        *col *= mult;
    }
    color
}

fn get_color(color: i32) -> Vec<f64> {

    let mut rng = rand::thread_rng();
    //let variation : f64 = rng.gen_range(-127.0, 127.0);

    /*let mut chosen_color = match color {
        0 => vec![127.0,  0.0,  0.0],
        1 => vec![0.0,  127.0,  0.0],
        2 => vec![0.0,  0.0,  127.0],
        3 => vec![127.0,  127.0, 0.0],
        4 => vec![127.0,  0.0, 127.0],
        5 => vec![0.0,  127.0, 127.0],
        _ => vec![0.0,  0.0, 0.0],
    };
*/
    /*for c in chosen_color.iter_mut() {
        if *c < 0.1 {
            *c += variation.abs()
        } else {
            *c += variation;
        }
    }*/
     
    vec![rng.gen_range(0.0, 255.0),rng.gen_range(0.0, 255.0),rng.gen_range(0.0, 255.0)]

    //chosen_color
}


use piston_window::*;
use sdl2_window::Sdl2Window;
fn initialize_window(width: u32, height: u32) -> PistonWindow {

    let window: PistonWindow = WindowSettings::new("Self Organizing Map", [width, height])
        .exit_on_esc(true)
        .build()
        .unwrap();
    
    window
}

extern crate image as im;
use im::ImageBuffer;
fn main() {

    let data_size = 1000000;

    let width = 600;
    let height = 600;

    let mut window = initialize_window(width, height);

    println!("Allocating data...");

    let mut data = Vec::<Vec<f64>>::with_capacity(data_size);    

    let mut rng = rand::thread_rng();
    
    for _ in 0 .. data_size {
        let color : i32 = rng.gen_range(0, 6);
        let color_buf = normalize(get_color(color), 255.0);
        data.push(color_buf);
    }
    
    println!("Initializing SOM");
    let mut som = SelfOrganizingMap::new(3, 10000, width as usize, height as usize);

    let mut canvas = ImageBuffer::new(width as u32, height as u32);

    let mut texture_context = TextureContext {
        factory: window.factory.clone(),
        encoder: window.factory.create_command_buffer().into()
    };

    let mut texture : G2dTexture = Texture::from_image(
        &mut texture_context, 
        &canvas,
        &TextureSettings::new()).unwrap();


    som.train(&data, |nodes: &[Node], epoch: u32| {

        if epoch % 50 == 0 {
         //   std::thread::sleep(std::time::Duration::from_millis(1));
            println!("Training epoch {:?}", epoch);
            let mut rendered = false;

            for node in nodes {        
                let as_rgb = normalize(node.get_weights().to_vec(), 1.0 / 255.0);

                canvas.put_pixel(node.x as u32, node.y as u32, im::Rgba([
                    as_rgb[0] as u8,
                    as_rgb[1] as u8,
                    as_rgb[2] as u8,
                    255
                ]));
            }

            texture.update(&mut texture_context, &canvas).unwrap();

            while !rendered {
                let mut event = window.next();

                while event.is_none() {
                    println!("Trying to get window event....");
                    event = window.next();
                }
                println!("Rendering...");
               
                let result = window.draw_2d(&event.unwrap(), |ctx, gl, device| {
                    texture_context.encoder.flush(device);
                    clear([1.0; 4], gl);
                    image(&texture, ctx.transform, gl);
                });
                
                println!("result = {:?}", result);

                if result.is_none() {
                    rendered = false;
                } else {
                    rendered = true;
                }


            }   
        }

    });
}
