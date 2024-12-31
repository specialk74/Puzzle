use anyhow::anyhow;
use itertools::Itertools;
use opencv::core::Point;
use opencv::core::Vector;
use opencv::{self as cv, prelude::*};
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use serde_json::{from_str, to_writer_pretty};
use std::collections::HashMap;
use std::fs;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use strum::IntoEnumIterator;

mod contour_with_dir;
mod cv_utils;
mod draw;
mod single_contour_params;

use crate::contour_with_dir::*;
use crate::cv_utils::*;
use crate::single_contour_params::SingleContourParams;

#[derive(Clone, Debug, Default)]
struct PuzzlePiece {
    file_name: String,
    contours: Vector<Vector<Point>>,
    contours_with_dir: Vec<ContourWithDir>,

    x_min: Point,
    y_min: Point,
    x_max: Point,
    y_max: Point,

    original_image: Mat,
    grey: Mat,
    original_contours: Vector<Vector<Point>>,
    corners: Vector<Point>,

    cx: i32,
    cy: i32,

    left_up_corner: Point,
    left_down_corner: Point,
    right_up_corner: Point,
    right_down_corner: Point,

    rect: cv::core::Rect,
    threshold: i32,
    center: Point,

    polygon: HashMap<Direction, Vector<Point>>,
    ok: bool,
    write_json: bool,
}

impl PuzzlePiece {
    fn new() -> Self {
        Self {
            file_name: "puzzle".to_string(),
            contours: Vector::new(),
            contours_with_dir: Vec::new(),

            x_min: Point::new(i32::MAX, 0),
            y_min: Point::new(0, i32::MAX),
            x_max: Point::new(0, 0),
            y_max: Point::new(0, 0),

            original_image: cv::core::Mat::default(),
            grey: cv::core::Mat::default(),
            original_contours: Vector::new(),
            corners: Vector::new(),

            cx: 0,
            cy: 0,

            left_up_corner: Point::new(i32::MAX, i32::MAX),
            left_down_corner: Point::new(i32::MAX, 0),
            right_up_corner: Point::new(0, i32::MAX),
            right_down_corner: Point::new(0, 0),

            rect: cv::core::Rect::default(),

            threshold: 0,
            center: Point::new(0, 0),
            polygon: HashMap::new(),
            ok: false,
            write_json: false,
        }
    }

    fn find_min_max(&mut self) -> std::io::Result<()> {
        for contour in &self.original_contours {
            for point in contour.iter() {
                if point.x < self.x_min.x {
                    self.x_min.x = point.x;
                    self.x_min.y = point.y;
                }

                if point.x > self.x_max.x {
                    self.x_max.x = point.x;
                    self.x_max.y = point.y;
                }

                if point.y < self.y_min.y {
                    self.y_min.x = point.x;
                    self.y_min.y = point.y;
                }

                if point.y > self.y_max.y {
                    self.y_max.x = point.x;
                    self.y_max.y = point.y;
                }
            }
        }

        Ok(())
    }

    fn set_corners(&mut self, corners: &Vector<Point>) {
        self.corners = corners.clone();

        for point in self.corners.iter() {
            if point.y < self.cy {
                if point.x < self.cx {
                    self.left_up_corner.x = point.x;
                    self.left_up_corner.y = point.y;
                } else {
                    self.right_up_corner.x = point.x;
                    self.right_up_corner.y = point.y;
                }
            } else if point.x < self.cx {
                self.left_down_corner.x = point.x;
                self.left_down_corner.y = point.y;
            } else {
                self.right_down_corner.x = point.x;
                self.right_down_corner.y = point.y;
            }
        }
    }

    fn search_best_threshold(&mut self) -> Result<(), anyhow::Error> {
        //println!("Search best threshold: {}", &self.file_name);

        let mut threshold = 0;
        let mut min_len = usize::MAX;
        for threshold_value in 160..230 {
            let (contours_cv, _phase) = match sub_process(&self.grey, &threshold_value) {
                Ok((im1, im2)) => (im1, im2),
                Err(err) => {
                    println!("search_best_threshold -> sub_process: {:?}", err);
                    return Err(anyhow!(err));
                }
            };

            let first = contours_cv.get(0).map_err(|_| anyhow!("Error"))?;

            if first.len() < min_len {
                min_len = first.len();
                threshold = threshold_value;
            }
        }

        // println!(
        //     "search_best_threshold -> {} -> min_len: {} - threshold: {}",
        //     self.file_name, min_len, threshold
        // );

        self.threshold = threshold;
        Ok(())
    }

    fn read_image(&mut self) -> Result<(), anyhow::Error> {
        //println!("OpenCV read file: {}", &self.file_name);
        self.original_image = read_image(&self.file_name)?;
        Ok(())
    }
}

fn main() {
    my_contour().unwrap();
}

fn find_files(path: &str) -> Vec<String> {
    fs::read_dir(path)
        .map(|entries| {
            entries
                .filter_map(Result::ok)
                .filter(|e| e.path().extension().is_some_and(|ext| ext == "jpg"))
                .filter_map(|e| e.path().to_str().map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_else(|_| Vec::new())
}

fn my_contour() -> Result<(), anyhow::Error> {
    let mut puzzles: Vec<PuzzlePiece> = find_files("./assets/")
        .into_par_iter()
        .map(|file_name| process(&file_name).unwrap_or_default())
        .collect();

    let mut puzzles_links = Vec::new();

    let mut last_file_name = String::new();
    for (element1, element2) in puzzles.iter().tuple_combinations() {
        if last_file_name != element1.file_name {
            last_file_name = element1.file_name.clone();
            println!("\n");
        }
        let (_, link1, link2) = match_shapes(element1, element2)?;
        puzzles_links.push(link1);
        puzzles_links.push(link2);
    }

    for puzzle in puzzles.iter_mut() {
        for puzzle_link in puzzles_links.iter_mut() {
            if puzzle.file_name == puzzle_link.file_name {
                for sequence in puzzle.contours_with_dir.iter_mut() {
                    for sequence_link in puzzle_link.contours_with_dir.iter() {
                        if !sequence_link.links.is_empty() && sequence.dir == sequence_link.dir {
                            for (k, v) in sequence_link.links.iter() {
                                sequence.links.insert(k.to_string(), *v);
                            }
                        }
                    }
                }
            }
        }

        if puzzle.ok && puzzle.write_json {
            let _ = to_writer_pretty(
                &File::create(format!("{}.json", puzzle.file_name))?,
                &puzzle.contours_with_dir,
            );
        }
    }

    Ok(())
}

fn process(file_name: &str) -> Result<PuzzlePiece, anyhow::Error> {
    println!("Process: {} ...", file_name);
    let mut puzzle = PuzzlePiece::new();
    puzzle.file_name = file_name.to_string();

    // region: Read json file
    // Check if json file exists. In case yes, load it.
    let path = format!("{}.json", puzzle.file_name);
    if Path::new(&path).exists() {
        let val = std::fs::read_to_string(path)?;
        let mut u: Vec<ContourWithDir> = from_str(&val)?;
        for contour_with_dir in u.iter_mut() {
            contour_with_dir.clear_links();
        }
        puzzle.contours_with_dir = u;
        return Ok(puzzle);
    }
    // endregion

    // region: Read image
    puzzle.read_image()?;
    // endregion

    // region: Convert to grey
    //show_image("Prima", &puzzle.original_image);
    println!("Converto to gray scale: {}", &puzzle.file_name);
    let phase = match to_grey(&puzzle.original_image) {
        Ok(phase) => phase,
        Err(err) => {
            println!("ToGrey Error: {:?}", err);
            return Err(anyhow!(err));
        }
    };
    //let _ = write_image(format!("{}_grey.jpg", puzzle.file_name).as_str(), &phase);
    // endregion

    // region: Convert to blur
    println!("Converto to blur: {}", &puzzle.file_name);
    puzzle.grey = match blur(&phase) {
        Ok(value) => value,
        Err(err) => {
            println!("Blur Error: {:?}", err);
            return Err(anyhow!(err));
        }
    };
    //let _ = write_image(format!("{}_blur.jpg", puzzle.file_name).as_str(), &puzzle.grey,);
    // endregion

    // region: Search best threshold
    puzzle.search_best_threshold()?;
    // endregion

    // region: Sub process
    println!("Sub process: {}", &puzzle.file_name);
    let (contours, phase) = match sub_process(&puzzle.grey, &puzzle.threshold) {
        Ok((im1, im2)) => (im1, im2),
        Err(err) => {
            println!("process->sub_process Error: {:?}", err);
            return Err(anyhow!(err));
        }
    };
    puzzle.original_contours = contours;
    // let _ = write_image(
    //     format!("{}_sub_process.jpg", puzzle.file_name).as_str(),
    //     &phase,
    // );
    // endregion

    // region: Find bounding rect
    println!("Find bounding rect: {}", &puzzle.file_name);
    puzzle.rect = match find_bounding_rect(&puzzle.contours, &phase) {
        Ok(image) => image,
        Err(err) => {
            println!("find_bounding_rect Error: {:?}", err);
            return Err(anyhow!(err));
        }
    };
    // endregion

    // region: Find centroid
    println!("Find centroid: {}", &puzzle.file_name);
    let Ok((cx, cy)) = find_centroid(&puzzle) else {
        todo!()
    };
    puzzle.cx = cx;
    puzzle.cy = cy;
    puzzle.center = Point::new(puzzle.cx, puzzle.cy);
    // endregion

    // region: Find min max
    println!("Find min max: {}", &puzzle.file_name);
    let _ = puzzle.find_min_max();
    // endregion

    // region: Fill poly
    println!("Fill poly: {}", &puzzle.file_name);
    let phase = match fill_poly(&puzzle) {
        Ok(image) => image,
        Err(err) => {
            println!("fill_poly Error: {:?}", err);
            return Err(anyhow!(err));
        }
    };
    // endregion

    // let _ = write_image(
    //     format!("{}_fill_poly.jpg", puzzle.file_name).as_str(),
    //     &phase,
    // );

    // region: Find corners
    println!("Find corners: {}", &puzzle.file_name);
    let corners;
    let (max1, corners1) = match find_corners(&puzzle, &puzzle.grey) {
        Ok((im1, im2)) => (im1, im2),
        Err(err) => {
            println!("process -> find_corners Error: {:?}", err);
            return Err(anyhow!(err));
        }
    };
    let (max2, corners2) = match find_corners(&puzzle, &phase) {
        Ok((im1, im2)) => (im1, im2),
        Err(err) => {
            println!("process -> find_corners Error: {:?}", err);
            return Err(anyhow!(err));
        }
    };

    if max1 > max2 {
        corners = corners1;
    } else {
        corners = corners2;
    }

    if corners.len() < 4 {
        println!(
            "process -> find_corners Error: Too low corners: {:?}",
            corners
        );
        return Err(anyhow!(
            "process -> find_corners Error: Too low corners: {:?}",
            corners
        ));
    }

    puzzle.set_corners(&corners);
    // endregion

    //let _ = draw_simple_contour(&puzzle);

    // region: Split contour
    println!("Split contour: {}", &puzzle.file_name);
    (puzzle.contours, puzzle.contours_with_dir) = match split_contour(&mut puzzle) {
        Ok((im1, im2)) => (im1, im2),
        Err(err) => {
            println!("split_contour Error: {:?}", err);
            return Err(anyhow!(err));
        }
    };
    // endregion

    // region: Draw contour
    println!("Draw contour: {}", &puzzle.file_name);
    //write_contour(&puzzle)?;
    match draw_contour(&puzzle) {
        Ok(image) => image,
        Err(err) => {
            println!("draw_contour Error: {:?}", err);
            return Err(anyhow!(err));
        }
    };
    // endregion

    puzzle.ok = true;
    Ok(puzzle)
}

fn fill_poly(puzzle: &PuzzlePiece) -> Result<Mat, anyhow::Error> {
    let mut new_phase = cv::core::Mat::new_size_with_default(
        puzzle.original_image.size()?,
        cv::core::CV_8UC1,
        get_black_color(),
    )?;

    match cv::imgproc::fill_poly_def(&mut new_phase, &puzzle.original_contours, get_white_color()) {
        Ok(_) => {}
        Err(err) => println!("Error on fill_convex_poly: {}", err),
    }

    let _name = format!("./{}fill_convex_poly.jpg", puzzle.file_name);
    //cv::imgcodecs::imwrite(&name, &new_phase, &cv::core::Vector::default())?;

    Ok(new_phase)
}

fn get_gender(
    puzzle: &PuzzlePiece,
    direction: Direction,
    contour: &Vector<Point>,
) -> Result<Genders, anyhow::Error> {
    let gender;

    let convex = cv::imgproc::bounding_rect(contour)?;
    // println!("{} - {:?} - bounding_rect: {:?}",puzzle.file_name, direction, convex);

    match direction {
        Direction::Down => {
            let max_corner = if puzzle.left_down_corner.y > puzzle.right_down_corner.y {
                puzzle.left_down_corner.y
            } else {
                puzzle.right_down_corner.y
            };

            if puzzle.y_max.y - max_corner > 100 {
                gender = Genders::Male;
            } else if convex.width < 200 || convex.height < 200 {
                gender = Genders::Line;
            } else {
                gender = Genders::Female;
            }
        }
        Direction::Left => {
            let max_corner = if puzzle.left_down_corner.x > puzzle.left_up_corner.x {
                puzzle.left_up_corner.x
            } else {
                puzzle.left_down_corner.x
            };

            if max_corner - puzzle.x_min.x > 100 {
                gender = Genders::Male;
            } else if convex.width < 200 || convex.height < 200 {
                gender = Genders::Line;
            } else {
                gender = Genders::Female;
            }
        }
        Direction::Right => {
            let max_corner = if puzzle.right_up_corner.x > puzzle.right_down_corner.x {
                puzzle.right_up_corner.x
            } else {
                puzzle.right_down_corner.x
            };

            if puzzle.x_max.x - max_corner > 100 {
                gender = Genders::Male;
            } else if convex.width < 200 || convex.height < 200 {
                gender = Genders::Line;
            } else {
                gender = Genders::Female;
            }
        }
        Direction::Up => {
            let max_corner = if puzzle.left_up_corner.y > puzzle.right_up_corner.y {
                puzzle.right_up_corner.y
            } else {
                puzzle.left_up_corner.y
            };

            if max_corner - puzzle.y_min.y > 100 {
                gender = Genders::Male;
            } else if convex.width < 200 || convex.height < 200 {
                gender = Genders::Line;
            } else {
                gender = Genders::Female;
            }
        }
    }
    Ok(gender)
}

fn split_contour(
    puzzle: &mut PuzzlePiece,
) -> Result<(Vector<Vector<Point>>, Vec<ContourWithDir>), anyhow::Error> {
    let mut contour_values = Vector::new();
    let mut contour_values_with_dir = Vec::new();

    // Per ogni direzione
    for dir in Direction::iter() {
        // Splitto il contour in base alla direzione
        println!("Split single contour: {} - {:?}", puzzle.file_name, dir);
        let SingleContourParams {
            single_contours,
            countour_traslated,
            x_max,
            y_min,
            y_max,
        } = match split_single_contour(puzzle, dir) {
            Ok(params) => params,
            Err(err) => {
                println!("Err split_contour -> split_single_contour {:?}", err);
                return Err(anyhow!(err));
            }
        };
        println!("Get Gender: {} - {:?}", puzzle.file_name, dir);
        let gender = match get_gender(puzzle, dir, &single_contours) {
            Ok(g) => g,
            Err(err) => {
                println!("Err split_contour -> get_gender: {:?}", err);
                return Err(anyhow!(err));
            }
        };
        println!("Create ContourWithDir: {} - {:?}", puzzle.file_name, dir);
        let mut c = ContourWithDir::new(
            single_contours.clone(),
            dir,
            gender,
            countour_traslated,
            x_max,
            y_min,
            y_max,
        );
        c.dx();
        contour_values_with_dir.push(c);
        contour_values.push(single_contours);
    }

    Ok((contour_values, contour_values_with_dir))
}

fn sub_process(
    grey_phase: &Mat,
    threshold_value: &i32,
) -> Result<(Vector<Vector<Point>>, Mat), anyhow::Error> {
    let phase = match threshold(grey_phase, *threshold_value) {
        Ok(im) => im,
        Err(err) => {
            println!("sub_process->threshold Err: {:?}", err);
            return Err(anyhow!(err));
        }
    };
    let phase = match bitwise(&phase) {
        Ok(im) => im,
        Err(err) => {
            println!("sub_process->bitwise Err: {:?}", err);
            return Err(anyhow!(err));
        }
    };
    let phase = match morph(&phase) {
        Ok(im) => im,
        Err(err) => {
            println!("sub_process->morph Err: {:?}", err);
            return Err(anyhow!(err));
        }
    };
    let contour_values = match find_contour(&phase) {
        Ok(im) => im,
        Err(err) => {
            println!("sub_process->find_contour Err: {:?}", err);
            return Err(anyhow!(err));
        }
    };

    //cv::highgui::imshow("sub_process", &phase);
    //cv::highgui::wait_key(500)?;

    Ok((contour_values, phase))
}

fn find_centroid(puzzle: &PuzzlePiece) -> Result<(i32, i32), anyhow::Error> {
    let mut cx = 0.0;
    let mut cy = 0.0;

    let first = puzzle
        .original_contours
        .get(0)
        .map_err(|_| anyhow!("Error"))?;

    match cv::imgproc::moments_def(&first) {
        Ok(moment) => {
            // println!("moment: {:?}", moment);
            // println!("X: {}, Y: {}", moment.m10/moment.m00, moment.m01/moment.m00);
            cx = moment.m10 / moment.m00;
            cy = moment.m01 / moment.m00;
        }
        Err(err) => println!("Error: {:?}", err),
    }

    Ok((cx as i32, cy as i32))
}

fn get_polygon(
    puzzle: &PuzzlePiece,
    delta: i32,
    direction: &Direction,
    iteration: i32,
) -> Vector<Point> {
    let mut polygon = Vector::new();

    match direction {
        Direction::Down => {
            let delta1 = puzzle.y_max.y + delta;
            polygon.push(puzzle.left_down_corner);
            polygon.push(Point::new(puzzle.left_down_corner.x, delta1));
            polygon.push(Point::new(puzzle.right_down_corner.x, delta1));
            polygon.push(puzzle.right_down_corner);
        }
        Direction::Up => {
            let delta1 = puzzle.y_min.y - delta;
            polygon.push(puzzle.left_up_corner);
            polygon.push(Point::new(puzzle.left_up_corner.x, delta1));
            polygon.push(Point::new(puzzle.right_up_corner.x, delta1));
            polygon.push(puzzle.right_up_corner);
        }
        Direction::Right => {
            let delta1 = puzzle.x_max.x + delta;
            polygon.push(puzzle.right_up_corner);
            polygon.push(Point::new(delta1, puzzle.right_up_corner.y));
            polygon.push(Point::new(delta1, puzzle.right_down_corner.y));
            polygon.push(puzzle.right_down_corner);
        }
        Direction::Left => {
            let delta1 = puzzle.x_min.x - delta;
            polygon.push(puzzle.left_up_corner);
            polygon.push(Point::new(delta1, puzzle.left_up_corner.y));
            polygon.push(Point::new(delta1, puzzle.left_down_corner.y));
            polygon.push(puzzle.left_down_corner);
        }
    }

    let valore = delta * iteration * if iteration % 2 == 0 { 1 } else { -1 };
    let mut mid = puzzle.center;
    match direction {
        Direction::Down | Direction::Up => {
            mid.y = puzzle.center.y + valore;
        }
        Direction::Left | Direction::Right => {
            mid.x = puzzle.center.x + valore;
        }
    }
    polygon.push(mid);

    println!(
        "Crea il poligono direction: {:?} - poligon: {:?} per il file {:?}",
        direction, polygon, &puzzle.file_name
    );
    polygon
}

fn split_single_contour(
    puzzle: &mut PuzzlePiece,
    direction: Direction,
) -> Result<SingleContourParams, anyhow::Error> {
    let mut vector = Vector::new();

    println!(
        "Recupero il primo contour per la direzione: {:?} del file {:?}",
        direction, &puzzle.file_name
    );
    let first = puzzle
        .original_contours
        .get(0)
        .map_err(|_| anyhow!("Error"))?;
    let mut onda;
    let mut count;
    for iteration in 0..100 {
        onda = 0;
        count = 0;
        vector.clear();
        // println!(
        //     "Iterazione #{} nella direzione {:?} per il file {:?}",
        //     iteration, direction, &puzzle.file_name
        // );
        // Crea il poligono che parte dal centro dell'immagine e arriva
        // alle 2 estremità nella direzione di direction;
        // In base ad iteration, modifica il poligono per cercare di recupeare
        // il lato del contour corretto
        let polygon = get_polygon(puzzle, 10, &direction, iteration);
        // Per ogni punto del contour
        for point in first.iter() {
            // Controllo se il punto è dentro il poligono che parte dal
            // centro dell'immagine e arriva ai due angoli estremi
            match cv::imgproc::point_polygon_test(
                &polygon,
                cv::core::Point2f::new(point.x as f32, point.y as f32),
                true,
            ) {
                Ok(val) => {
                    if val > 0.0 {
                        // Il punto è dentro il poligono
                        if onda != 1 {
                            onda = 1;
                            count += 1;
                        }
                        vector.push(Point::new(point.x, point.y));
                    } else if onda != 2 {
                        // Il punto è fuori dal poligono
                        onda = 2;
                        count += 1;
                    }
                }
                Err(err) => {
                    println!("Error on split_single_contour: {}", err);
                }
            }
        }

        // let _ = draw_contour2(puzzle, &polygon);

        // Se ogni punto del contorno ha fatto meno di 3 entri/esci dal poligono, lo prendo per buono
        if count <= 3 {
            puzzle.polygon.insert(direction, polygon.clone());
            break;
        }
    }

    let (up_left_point, _down_right_point) = match get_extreme(direction, &vector) {
        Ok((im1, im2)) => (im1, im2),
        Err(err) => {
            println!(
                "Err split_single_contour -> get_extreme(up_left_point, _down_right_point): {:?}",
                err
            );
            return Err(anyhow!(err));
        }
    };

    let mut vector_after_first_traslate = Vector::new();
    for point in vector.iter() {
        vector_after_first_traslate.push(point - up_left_point);
    }

    let (up_left_point_translated, down_right_point_translated) = match get_extreme(
        direction,
        &vector_after_first_traslate,
    ) {
        Ok((im1, im2)) => (im1, im2),
        Err(err) => {
            println!("Err split_single_contour -> get_extreme(up_left_point_translated, down_right_point_translated): {:?}", err);
            return Err(anyhow!(err));
        }
    };
    let mut angle =
        (down_right_point_translated.y as f64 / down_right_point_translated.x as f64).atan();

    if angle < 0.0 {
        if down_right_point_translated.y < 0 {
            angle = (2.0 * std::f64::consts::PI + angle).abs();
        } else {
            angle = (std::f64::consts::PI + angle).abs();
        }
    }

    let m = cv::imgproc::get_rotation_matrix_2d(
        cv::core::Point2f::new(
            up_left_point_translated.x as f32,
            up_left_point_translated.y as f32,
        ),
        angle.to_degrees(),
        1.0,
    )?;

    let mut vector_rotated = vector_after_first_traslate.clone();

    match cv::core::transform(&vector_after_first_traslate, &mut vector_rotated, &m) {
        Ok(_) => {}
        Err(err) => {
            println!(
                "Err split_single_contour -> transform: {:?} - m: {:#?}",
                err, m
            );
            return Err(anyhow!(err));
        }
    };

    let mut y_max = 0;
    for point in vector_rotated.iter() {
        if y_max < point.y {
            y_max = point.y;
        }
    }

    let mut vector_traslated = Vector::new();

    if y_max > 100 {
        for point in vector_rotated.iter() {
            vector_traslated.push(Point::new(point.x, -point.y));
        }
    } else {
        vector_traslated = vector_rotated.clone();
    }

    let mut x_max = 0;
    let mut y_min = i32::MAX;
    let mut y_max = 0;
    let mut x_min = i32::MAX;

    for point in vector_traslated.iter() {
        if point.x < x_min {
            x_min = point.x;
        }
        if point.x > x_max {
            x_max = point.x;
        }
        if point.y < y_min {
            y_min = point.y;
        }
        if point.y > y_max {
            y_max = point.y;
        }
    }

    //let _ = cv::highgui::imshow("Phase", &phase);
    //draw_simple_contour(puzzle)?;
    //let key = cv::highgui::wait_key(500)?;
    //draw_contour(puzzle);

    Ok(SingleContourParams {
        single_contours: vector,
        countour_traslated: vector_traslated,
        x_max,
        y_min,
        y_max,
    })
}

fn find_corners_gui(
    puzzle: &PuzzlePiece,
    phase: &Mat,
) -> Result<(f64, Vector<Point>), anyhow::Error> {
    let mut corners = Vector::new();
    let mut max_corners = 4;
    let mut quality_level = 0.1;
    let mut min_distance = 1300.0;
    let mut block_size: i32 = 100;
    let use_harris_detector: bool = false;
    let mut k: f64 = 0.1;

    let mut last_min_distance = 0.0;
    let mut last_block_size = 0;
    let mut last_quality_level = 0.0;
    let mut last_max_corners = 0;
    let mut last_k = 0.0;

    // let _ = cv::highgui::imshow("Phase", &phase);

    loop {
        if last_min_distance != min_distance
            || last_block_size != block_size
            || last_quality_level != quality_level
            || last_max_corners != max_corners
            || last_k != k
        {
            last_min_distance = min_distance;
            last_block_size = block_size;
            last_quality_level = quality_level;
            last_max_corners = max_corners;
            last_k = k;

            println!(
                "min_distance: {} - block_size: {} - quality_level: {} - max_corners: {} - k: {}",
                min_distance, block_size, quality_level, max_corners, k
            );

            match cv::imgproc::good_features_to_track(
                &phase,
                &mut corners,
                max_corners,
                quality_level,
                min_distance,
                &cv::core::no_array(),
                block_size,
                use_harris_detector,
                k,
            ) {
                Ok(_) => {}
                Err(err) => println!(
                    "Error on find_corners (block_size {}): {} with ",
                    block_size, err
                ),
            }

            // let value_min_area = cv::imgproc::min_area_rect(&corners)?;
            // let mut points = VectorOfPoint::default();
            // let point1 = Point::new(value_min_area.center.x as i32, value_min_area.center.y as i32);
            // let point2 = Point::new(puzzle.cx, puzzle.cy);
            // let diff = point1 - point2;
            // points.push(diff);
            // let norm = cv::core::norm_def(&points)?;

            // dbg!(norm);

            // let norm = cv::core::norm_def(&corners)?;
            // dbg!(&norm);

            let mut new_phase = puzzle.original_image.clone();

            for point in corners.iter() {
                cv::imgproc::circle(
                    &mut new_phase,
                    point,
                    20,
                    cv::core::Scalar::new(0.0, 0.0, 255.0, 255.0),
                    cv::imgproc::FILLED,
                    cv::imgproc::LINE_8,
                    0,
                )?;
            }

            // let zero_offset = Point::new(0, 0);
            // let thickness: i32 = 20;

            // cv::imgproc::draw_contours(&mut new_phase,
            //     &puzzle.original_contours,
            //     -1,
            //     color,
            //     thickness,
            //     cv::imgproc::LINE_8,
            //     &cv::core::no_array(),
            //     2,
            //     zero_offset)?;

            let _ = cv::highgui::imshow("Corner", &new_phase);
        }

        let key = cv::highgui::wait_key(500)?;

        match key {
            27 => break,
            66 => block_size += 5, // B
            98 => block_size -= 5, // b

            51 => min_distance += 1.0, // 1
            52 => min_distance -= 1.0, // 2

            68 => min_distance += 10.0,  // D
            100 => min_distance -= 10.0, // d

            81 => quality_level += 0.1,  // Q
            113 => quality_level -= 0.1, // q

            67 => max_corners += 1, // C
            99 => max_corners -= 1, // c

            75 => k += 0.1,  // K
            107 => k -= 0.1, // k
            _ => {
                if key != -1 {
                    println!("Key: {}", key);
                }
            }
        }
    }

    Ok((0.0, corners))
}

fn find_corners(_puzzle: &PuzzlePiece, phase: &Mat) -> Result<(f64, Vector<Point>), anyhow::Error> {
    let mut corners = Vector::new();
    let max_corners = 4;
    let quality_level = 0.1;
    let mut distance = 700.0;
    let mut block_size;
    let use_harris_detector: bool = true;
    let k: f64 = 0.1;
    let mut min_corners = Vector::new();
    let mut points: Vector<Point> = Vector::new();
    let contour_center = Point::new(_puzzle.cx, _puzzle.cy);
    let mut max_tot_distance = 0.0;
    loop {
        block_size = 60;
        loop {
            match cv::imgproc::good_features_to_track(
                &phase,
                &mut corners,
                max_corners,
                quality_level,
                distance,
                &cv::core::no_array(),
                block_size,
                use_harris_detector,
                k,
            ) {
                Ok(_) => {}
                Err(err) => println!(
                    "Error on find_corners (block_size {}): {} with ",
                    block_size, err
                ),
            };

            if corners.len() == 4 {
                let value_min_area = cv::imgproc::min_area_rect(&corners)?;

                points.clear();
                let min_area_center = Point::new(
                    value_min_area.center.x as i32,
                    value_min_area.center.y as i32,
                );
                let diff = min_area_center - contour_center;
                points.push(diff);

                let p1_diff = corners.get(0)? - contour_center;
                points.clear();
                points.push(p1_diff);

                let p2_diff = corners.get(1)? - contour_center;
                points.clear();
                points.push(p2_diff);

                let p3_diff = corners.get(2)? - contour_center;
                points.clear();
                points.push(p3_diff);

                let p4_diff = corners.get(3)? - contour_center;
                points.clear();
                points.push(p4_diff);

                points.push(p1_diff);
                points.push(p2_diff);
                points.push(p3_diff);
                let tot_distance = cv::core::norm_def(&points)?;

                if max_tot_distance < tot_distance {
                    min_corners = corners.clone();
                    max_tot_distance = tot_distance;
                }
            }

            block_size += 20;
            if block_size > 90 {
                break;
            }
        }
        distance += 20.0;
        if distance > 1000.0 {
            break;
        }
    }

    Ok((max_tot_distance, min_corners))
}

fn write_contour(puzzle: &PuzzlePiece) -> std::io::Result<()> {
    let mut output = String::new();
    for contours_with_dir in &puzzle.contours_with_dir {
        output += &format!("Direction: {:?}\n", contours_with_dir.dir);
        output += &format!("Gender: {:?}\n", contours_with_dir.gender);
        for point in contours_with_dir.countour.iter() {
            output += &format!("{},{}\n", point.x, point.y);
        }
        output += "\n\n";
    }

    let mut file = File::create(format!("{}_contour.txt", puzzle.file_name))?;
    file.write_all(output.as_bytes())?;

    Ok(())
}

fn draw_contour(puzzle: &PuzzlePiece) -> Result<(), anyhow::Error> {
    let mut phase = puzzle.original_image.clone();

    //draw_traslated_contour(puzzle, &mut phase)?;

    //draw_area_inside_polygon(puzzle, &mut phase)?;

    draw::draw_four_angles(&puzzle.corners, &mut phase)?;

    draw::draw_internal_contour(&puzzle.contours, &mut phase)?;

    println!("Salva l'immagine del file {:?}", puzzle.file_name);
    let _ = write_image(
        format!("{}_contours.jpg", puzzle.file_name).as_str(),
        &phase,
    );
    //let _ = cv::highgui::imshow("Phase", &phase);
    //let _ = cv::highgui::wait_key(0)?;

    Ok(())
}

fn match_shapes(
    puzzle1_orig: &PuzzlePiece,
    puzzle2_orig: &PuzzlePiece,
) -> Result<(bool, PuzzlePiece, PuzzlePiece), anyhow::Error> {
    let mut puzzle1 = puzzle1_orig.clone();
    let mut puzzle2 = puzzle2_orig.clone();
    let mut add = false;
    for sequence1 in puzzle1.contours_with_dir.iter_mut() {
        for sequence2 in puzzle2.contours_with_dir.iter_mut() {
            if sequence1.gender != Genders::Line
                && sequence2.gender != Genders::Line
                && sequence1.gender != sequence2.gender
            {
                //sequence1.dx();
                //sequence2.dx();
                let d1 = (sequence1.d1 - sequence2.d1).abs();
                let d2 = (sequence1.d2 - sequence2.d2).abs();
                let d3 = (sequence1.d3 - sequence2.d3).abs();
                let d4 = (sequence1.d4 - sequence2.d4).abs();
                let d5 = (sequence1.d5 - sequence2.d5).abs();
                let d1_d4 = (sequence1.d1 - sequence2.d4).abs();
                let d4_d1 = (sequence1.d4 - sequence2.d1).abs();
                if d5 < 100 && d3 < 30 && ((d1 < 60 && d4 < 60) || (d1_d4 < 60 && d4_d1 < 60)) {
                    println!("{} - {} - {:?} - {:?}; d1: {}; d2: {}; d3: {}, d4: {}; d5: {}; d1-d4: {}; d4-d1: {}",
                        &puzzle1.file_name, &puzzle2.file_name, sequence1.dir, sequence2.dir,
                        d1,
                        d2,
                        d3,
                        d4,
                        d5,
                        d1_d4,
                        d4_d1,
                    );
                    add = true;
                    //sequence1.links.push(format!("{}-{:?}",puzzle2.file_name, sequence2.dir));
                    sequence1
                        .links
                        .insert(puzzle2.file_name.clone(), sequence2.dir);
                    //sequence2.links.push(format!("{}-{:?}",puzzle1.file_name, sequence1.dir));
                    sequence2
                        .links
                        .insert(puzzle1.file_name.clone(), sequence1.dir);
                };
            }
        }
    }

    Ok((add, puzzle1, puzzle2))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::{self, File};
    use std::io::Write;
    use std::path::Path;

    #[test]
    fn test_find_files() {
        // Setup: create a temporary directory with some test files
        let test_dir = "test_dir";
        fs::create_dir(test_dir).unwrap();
        let test_files = vec!["test1.jpg", "test2.jpg", "test3.txt"];
        for file in &test_files {
            let file_path = Path::new(test_dir).join(file);
            let mut file = File::create(&file_path).unwrap();
            writeln!(file, "test content").unwrap();
        }

        // Call the function
        let result = find_files(test_dir);

        // Cleanup: remove the temporary directory and its contents
        fs::remove_dir_all(test_dir).unwrap();

        // Assert: check that the result contains only the .jpg files
        assert_eq!(result.len(), 2);
        assert!(result.contains(&format!("{}/test1.jpg", test_dir)));
        assert!(result.contains(&format!("{}/test2.jpg", test_dir)));
    }
}
