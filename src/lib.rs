use rand::prelude::*;

pub fn sigmoid(x: f32) -> f32 {
    1. / (1. + f32::exp(-x))
}

pub fn rand_float() -> f32 {
    let mut rng = rand::thread_rng();
    rng.gen()
}
