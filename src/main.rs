use gates::{rand_float, sigmoid};

const OR_TRAIN: [[f32; 3]; 4] = [[0., 0., 0.], [1., 0., 1.], [0., 1., 1.], [1., 1., 1.]];

const AND_TRAIN: [[f32; 3]; 4] = [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [1., 1., 1.]];

const NAND_TRAIN: [[f32; 3]; 4] = [[0., 0., 1.], [1., 0., 1.], [0., 1., 1.], [1., 1., 0.]];

fn cost(w1: f32, w2: f32, b: f32, data: [[f32; 3]; 4]) -> f32 {
    let mut result = 0.;

    for case in data {
        let x1 = case[0];
        let x2 = case[1];
        let y = sigmoid(x1 * w1 + x2 * w2 + b);
        let d = y - case[2];
        result += d * d;
    }

    result /= data.len() as f32;
    result
}

fn dcost(
    eps: f32,
    w1: f32,
    w2: f32,
    bias: f32,
    dw1: &mut f32,
    dw2: &mut f32,
    db: &mut f32,
    data: [[f32; 3]; 4],
) {
    let c = cost(w1, w2, bias, data);
    *dw1 = (cost(w1 + eps, w2, bias, data) - c) / eps;
    *dw2 = (cost(w1, w2 + eps, bias, data) - c) / eps;
    *db = (cost(w1, w2, bias + eps, data) - c) / eps;
}

fn train(data: [[f32; 3]; 4]) -> (f32, f32, f32) {
    let mut w1 = rand_float();
    let mut w2 = rand_float();
    let mut bias = rand_float();

    let eps = 1e-1;
    let rate = 1e-1;

    for _ in 0..1_000_000 {
        // let c = cost(w1, w2, bias, data);
        // println!("c = {}, w1 = {}, w2 = {}, b = {}", c, w1, w2, b);

        let mut dw1: f32 = 0.;
        let mut dw2: f32 = 0.;
        let mut db: f32 = 0.;

        dcost(eps, w1, w2, bias, &mut dw1, &mut dw2, &mut db, data);
        // gcost(w1, w2, b, &dw1, &dw2, &db);
        w1 -= rate * dw1;
        w2 -= rate * dw2;
        bias -= rate * db;
    }

    println!(
        "c = {}, w1 = {}, w2 = {}, b = {}",
        cost(w1, w2, bias, data),
        w1,
        w2,
        bias
    );

    (w1, w2, bias)
}

fn main() {
    for (name, data) in [("AND", AND_TRAIN), ("OR", OR_TRAIN), ("NAND", NAND_TRAIN)] {
        println!("{name}");
        let (w1, w2, bias) = train(data);
        for i in 0..2 {
            for j in 0..2 {
                println!(
                    "{} | {} = {}",
                    i,
                    j,
                    sigmoid(i as f32 * w1 + j as f32 * w2 + bias)
                );
            }
        }
        println!();
    }
}
