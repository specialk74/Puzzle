use anyhow::Error;
use anyhow::anyhow;
use anyhow::Result;
use opencv::{self as cv, prelude::*};
use serde::{Serialize, Deserialize};
use serde_json::{to_string_pretty, to_writer_pretty, from_str};
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use opencv::core::Point;
use rand::Rng;
use strum::{EnumIter, IntoEnumIterator};
use opencv::types::VectorOfPoint;
use opencv::types::VectorOfVectorOfPoint;
use itertools::Itertools;
use std::path::Path;

#[derive(EnumIter, Debug, Hash, Eq, PartialEq, Clone, Copy, Serialize, Deserialize)]
enum Direction {
    UpSide,
    DownSide,
    RightSide,
    LeftSide
}

#[derive(EnumIter, Debug, Hash, Eq, PartialEq, Clone, Copy, Serialize, Deserialize)]
enum Genders {
    Unknown,
    Female,
    Male,
    Line
}

#[derive(Debug, Serialize, Deserialize)]
struct ContourWithDir {
    #[serde(skip)]
    countour: VectorOfPoint,
    #[serde(skip)]
    countour_traslated: VectorOfPoint,
    dir: Direction,
    gender: Genders,
    x_max: i32,
    y_min: i32,
    y_max: i32,
    d1: i32,
    d2: i32,
    d3: i32,
    d4: i32,
    d5: i32,
}

struct PuzzlePiece {
    file_name: String,
    contours: VectorOfVectorOfPoint,
    contours_with_dir: Vec<ContourWithDir>,

    x_min: Point,
    y_min: Point,
    x_max: Point,
    y_max: Point,

    original_image: Mat,
    grey: Mat,
    original_contours: VectorOfVectorOfPoint,
    corners: VectorOfPoint,

    cx: i32,
    cy: i32,

    left_up_corner: Point,
    left_down_corner: Point,
    right_up_corner: Point,
    right_down_corner: Point,

    rect: cv::core::Rect,
    threshold: i32,
    center: Point,

    polygon: HashMap<Direction, VectorOfPoint>,
    matched: bool,
}

impl PuzzlePiece {
    fn new() -> Self {
        Self {
            file_name: "puzzle".to_string(),
            contours: VectorOfVectorOfPoint::default(),
            contours_with_dir: Vec::new(),

            x_min: Point::new(i32::MAX,0),
            y_min: Point::new(0,i32::MAX),
            x_max: Point::new(0,0),
            y_max: Point::new(0,0),

            original_image: cv::core::Mat::default(),
            grey: cv::core::Mat::default(),
            original_contours: VectorOfVectorOfPoint::default(),
            corners: VectorOfPoint::default(),

            cx : 0,
            cy: 0,

            left_up_corner: Point::new(i32::MAX,i32::MAX),
            left_down_corner: Point::new(i32::MAX,0),
            right_up_corner: Point::new(0,i32::MAX),
            right_down_corner: Point::new(0,0),

            rect: cv::core::Rect::default(),

            threshold: 0,
            center: Point::new(0,0),
            polygon: HashMap::new(),

            matched: false,
        }
    }
}

/* 
impl Serialize for ContourWithDir {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // 3 is the number of fields in the struct.
        let mut state = serializer.serialize_struct("ContourWithDir", 7)?;
        state.serialize_field("dir", &self.dir)?;
        state.serialize_field("gender", &self.gender)?;
        state.serialize_field("d1", &self.d1)?;
        state.serialize_field("d2", &self.d2)?;
        state.serialize_field("d3", &self.d3)?;
        state.serialize_field("d4", &self.d4)?;
        state.serialize_field("d5", &self.d5)?;
        state.end()
    }
}
*/

impl ContourWithDir {
    fn new (countour: VectorOfPoint, dir: Direction, gender: Genders, countour_traslated: VectorOfPoint, x_max: i32, y_min: i32, y_max: i32) -> Self {
        Self { 
            countour,
            dir,
            gender,
            countour_traslated,
            x_max,
            y_min, 
            y_max,
            d3: i32::MAX,
            d1: -1,
            d4: -1,
            d5: -1,
            d2: -1,
        }
    }

    fn dx(&mut self) {
        if self.d1 != -1 {
            return;
        }

        for point in self.countour_traslated.iter() {
            if point.y < self.d3 {
                self.d1 = point.x;
                self.d3 = point.y;
            }
            if point.x > self.d5 {
                self.d5 = point.x;
            }
        }
        self.d4 = self.d5 - self.d1;
        self.d2 = 0;

        for i in self.d3..-100 {
            let mut v = Vec::new();
            for point in self.countour_traslated.iter() {
                if point.y == i {
                    v.push(point.x);
                }
            }
            if v.len() > 1 {
                let mut x_min = i32::MAX;
                let mut x_max = 0;
                for x in v {
                    if x_min > x {
                        x_min = x;
                    }
                    if x_max < x {
                        x_max = x;
                    }
                }
                let diff = x_max - x_min;
                if diff > self.d2 {
                    self.d2 = diff;
                }
            }
        }
    }
}

fn main() -> Result<()> {
    my_contour()?;
Ok(())
}

fn my_contour() -> Result<(), anyhow::Error> {
    let file_names = vec![
    "IMG20240113121213", 
    "IMG20240113121228", 
    "IMG20240113121241",
    "IMG20240113121256",
    "IMG20240113121307",
    "IMG20240113121330",
    "IMG20240113121341",
    "IMG20240113121354",
    "IMG20240113121410",
    "IMG20240113121426",
    "IMG20240113121438",
    "IMG20240113121458",
    "IMG20240113121510",
    ];

    let puzzles: Vec<PuzzlePiece> = file_names
        .into_par_iter()
        .map(|file_name| { 
            process(file_name).unwrap()
        })
        .collect();
    // file_names.into_iter().for_each(|file_name| { process(file_name); });

    for puzzle in puzzles.iter() {
        if !Path::new(&format!("{}.json",puzzle.file_name)).exists() {
            let _ = to_writer_pretty(&File::create(format!("{}.json", puzzle.file_name))?, &puzzle.contours_with_dir);
        }
    }

    for (element1, element2) in puzzles.iter().tuple_combinations() {
        match_shapes(&element1, &element2)?;
    }

    Ok(())
}

fn process(file_name: &str) -> Result<PuzzlePiece, anyhow::Error> {
    println!("Process: {} ...",file_name);
    let mut puzzle = PuzzlePiece::new();
    puzzle.file_name = file_name.to_string();
    let path = format!("{}.json",puzzle.file_name);

    if Path::new(&path).exists() {
        let val = std::fs::read_to_string(path)?;
        let u: Vec<ContourWithDir> = from_str(&val)?;
        puzzle.contours_with_dir = u;
        return Ok(puzzle);
    }
    puzzle.original_image = cv::imgcodecs::imread(format!("./assets/{}.jpg", puzzle.file_name).as_str(), cv::imgcodecs::IMREAD_COLOR)?;
    
    //show_image("Prima", &puzzle.original_image);

    let phase = to_grey(&puzzle.original_image)?;
    let grey_phase = blur(&phase)?;
    puzzle.grey = grey_phase;

    let _pre_corner = pre_corner_detect_def(&puzzle)?;

    let threshold = search_best_threshold(&puzzle.grey)?;
    let (contours, phase) = sub_process(&puzzle.grey, threshold)?;
    puzzle.original_contours = contours;
    puzzle.threshold = threshold;

    puzzle.rect = find_bounding_rect(&puzzle, &phase, &puzzle.original_contours)?;

    let Ok((cx, cy)) = find_centroid(&puzzle) else { todo!() };
    puzzle.cx = cx;
    puzzle.cy = cy;
    puzzle.center = Point::new(puzzle.cx, puzzle.cy);

    let _ = find_min_max(&mut puzzle);

    let phase = fill_poly(&puzzle)?;

    // corner_herris_application(&phase);


    let corners;
    let (max1, corners1) = find_corners(&puzzle, &puzzle.grey)?;
    let (max2, corners2) = find_corners(&puzzle, &phase)?;

    if max1 > max2 {
        corners = corners1;
    }
    else {
        corners = corners2;
    }
    
    set_corners(&mut puzzle, &corners);

    puzzle.corners = corners;


    let _ = draw_simple_contour(&puzzle);

    (puzzle.contours, puzzle.contours_with_dir) = split_contour(&mut puzzle)?;

    //write_contour(&puzzle)?;
    draw_contour(&puzzle)?;
    
    Ok(puzzle)
}

fn pre_corner_detect_def(puzzle: &PuzzlePiece) -> Result<Mat, anyhow::Error> {
    let mut phase = puzzle.grey.clone();
    let ksize = 1;

    match cv::imgproc::pre_corner_detect_def(
        &puzzle.grey,
        &mut phase,
        ksize
    ){
        Ok(()) => {},
        Err(err) => {
            println!("Error during pre_corner_detect_def: {}", err);
            return Err(anyhow!(err));
        }
    }

    //show_image("pre_corner_detect_def", &phase);
    //wait_key(0);
    let phase = puzzle.grey.clone();

    Ok(phase)
}

fn wait_key(delay: i32) -> Result<i32,opencv::Error> {
    cv::highgui::wait_key(delay)
}

fn set_corners(puzzle: &mut PuzzlePiece, corners: &VectorOfPoint) {
    for point in corners.iter() {
        if point.y < puzzle.cy {
            if point.x < puzzle.cx {
                puzzle.left_up_corner.x = point.x;
                puzzle.left_up_corner.y = point.y;
            }
            else {
                puzzle.right_up_corner.x = point.x;
                puzzle.right_up_corner.y = point.y;
            }
        }
        else {
            if point.x < puzzle.cx {
                puzzle.left_down_corner.x = point.x;
                puzzle.left_down_corner.y = point.y;
            }
            else {
                puzzle.right_down_corner.x = point.x;
                puzzle.right_down_corner.y = point.y;
            }
        }
    }
}

fn get_black_color() -> cv::core::Scalar {
    cv::core::Scalar::new(0.0, 0.0, 0.0, 255.0)
}

fn get_white_color() -> cv::core::Scalar {
    cv::core::Scalar::new(255.0, 255.0, 255.0, 255.0)
}

fn fill_poly(puzzle: &PuzzlePiece)-> Result<Mat, anyhow::Error> {
    let mut new_phase = cv::core::Mat::new_size_with_default(
        puzzle.original_image.size()?,
        cv::core::CV_8UC1,
        get_black_color(), 
    )?;

    match cv::imgproc::fill_poly_def(
        &mut new_phase,
        &puzzle.original_contours,
        get_white_color(),
    ){
        Ok(_) => {},
        Err(err) => println!("Error on fill_convex_poly: {}", err)
    }

    let _name = format!("./{}fill_convex_poly.jpg", puzzle.file_name);
    //cv::imgcodecs::imwrite(&name, &new_phase, &cv::core::Vector::default())?;

    Ok(new_phase)
}

fn distance_between_points(point1: &Point, point2: &Point) -> Result<u32, anyhow::Error> {
    let mut points = VectorOfPoint::default();
    points.push(point1.clone());
    points.push(point2.clone());

    let num1 = (point1.x - point2.x).abs().pow(2) as f64;
    let num2 = (point1.y - point2.y).abs().pow(2) as f64;
    let num3 = (num1 + num2).sqrt();
    //dbg!(num3);
    //println!("Diff: {:?}", ((point1.x - point2.x).abs().pow(2.0) + (point1.y - point2.y).abs().pow(2.0)).sqrt());

    //Ok(cv::core::norm_def(&points)?)
    Ok(num3 as u32)
}

fn get_gender(puzzle: &PuzzlePiece, direction: Direction, contour: &VectorOfPoint) -> Result<Genders, Error> {
    let gender;

    let convex = cv::imgproc::bounding_rect(contour)?;
    // println!("{} - {:?} - bounding_rect: {:?}",puzzle.file_name, direction, convex);

    match direction {
        Direction::DownSide => {
            let max_corner;
            if puzzle.left_down_corner.y > puzzle.right_down_corner.y {
                max_corner = puzzle.left_down_corner.y;
            }
            else {
                max_corner = puzzle.right_down_corner.y;
            }

            if puzzle.y_max.y - max_corner > 100 {
                gender = Genders::Male;
            }
            else if convex.width < 200 || convex.height < 200 {
                gender = Genders::Line;
                // let max_value = vec![
                //     distance_between_points(&puzzle.left_down_corner, &Point::new(convex.x, convex.y))?,
                //     distance_between_points(&puzzle.left_down_corner, &Point::new(convex.x, convex.y + convex.height))?,
                //     distance_between_points(&puzzle.right_down_corner, &Point::new(convex.x + convex.width, convex.y))?,
                //     distance_between_points(&puzzle.right_down_corner, &Point::new(convex.x + convex.width, convex.y + convex.height))?
                //     ];
                //     dbg!(&max_value);
                //     match max_value.iter().max() {
                //     Some(val) => {
                //         if val > &150 {
                //             gender = Genders::Female;
                //         }
                //         else {
                //             gender = Genders::Line;
                //         }
                //     },
                //     None => {}
                // }
            }
            else {
                gender = Genders::Female;
            }


/*             println!("{} - direction: {:?} - LD: {:?} - RD: {:?} - gender: {:?} - diff: {:?} - {:?} - {:?} - {:?}", puzzle.file_name, direction,
            puzzle.left_down_corner,
            puzzle.right_down_corner,
            gender,
            distance_between_points(&puzzle.left_down_corner, &Point::new(convex.x, convex.y)),
            distance_between_points(&puzzle.left_down_corner, &Point::new(convex.x, convex.y + convex.height)),
            distance_between_points(&puzzle.right_down_corner, &Point::new(convex.x + convex.width, convex.y)),
            distance_between_points(&puzzle.right_down_corner, &Point::new(convex.x + convex.width, convex.y + convex.height))
            ); */
        },
        Direction::LeftSide => {
            let max_corner;
            if puzzle.left_down_corner.x > puzzle.left_up_corner.x {
                max_corner = puzzle.left_up_corner.x;
            }
            else {
                max_corner = puzzle.left_down_corner.x;
            }

            if max_corner - puzzle.x_min.x > 100 {
                gender = Genders::Male;
            }
            else if convex.width < 200 || convex.height < 200 {
                gender = Genders::Line;

                // let max_value = vec![
                //     distance_between_points(&puzzle.left_up_corner, &Point::new(convex.x, convex.y))?,
                //     distance_between_points(&puzzle.left_up_corner, &Point::new(convex.x+ convex.width, convex.y))?,
                //     distance_between_points(&puzzle.left_down_corner, &Point::new(convex.x, convex.y + convex.height))?,
                //     distance_between_points(&puzzle.left_down_corner, &Point::new(convex.x + convex.width, convex.y + convex.height))?,
                //     ];
                //     dbg!(&max_value);

                //     match max_value.iter().max() {
                //     Some(val) => {
                //         if val > &150 {
                //             gender = Genders::Female;
                //         }
                //         else {
                //             gender = Genders::Line;
                //         }
                //     },
                //     None => {}
                // }
            }
            else {
                gender = Genders::Female;
            }

            /*             println!("{} - direction: {:?} - LD: {:?} - LU: {:?} - gender: {:?} - diff: {:?} - {:?} - {:?} - {:?}", puzzle.file_name, direction,
            puzzle.left_down_corner,
            puzzle.left_up_corner,
            gender,
            distance_between_points(&puzzle.left_up_corner, &Point::new(convex.x, convex.y)),
            distance_between_points(&puzzle.left_up_corner, &Point::new(convex.x+ convex.width, convex.y)),
            distance_between_points(&puzzle.left_down_corner, &Point::new(convex.x, convex.y + convex.height)),
            distance_between_points(&puzzle.left_down_corner, &Point::new(convex.x + convex.width, convex.y + convex.height)),
            ); */
        },
        Direction::RightSide => {
            let max_corner;
            if puzzle.right_up_corner.x > puzzle.right_down_corner.x {
                max_corner = puzzle.right_up_corner.x;
            }
            else {
                max_corner = puzzle.right_down_corner.x;
            }

            if puzzle.x_max.x - max_corner > 100 {
                gender = Genders::Male;
            }
            else if convex.width < 200 || convex.height < 200 {
                gender = Genders::Line;
                // let max_value = vec![
                //     distance_between_points(&puzzle.right_up_corner, &Point::new(convex.x, convex.y))?,
                //     distance_between_points(&puzzle.right_up_corner, &Point::new(convex.x+ convex.width, convex.y))?,
                //     distance_between_points(&puzzle.right_down_corner, &Point::new(convex.x, convex.y + convex.height))?,
                //     distance_between_points(&puzzle.right_down_corner, &Point::new(convex.x + convex.width, convex.y + convex.height))?
                //     ];
                //     dbg!(&max_value);

                //     match max_value.iter().max() {
                //     Some(val) => {
                //         if val > &150 {
                //             gender = Genders::Female;
                //         }
                //         else {
                //             gender = Genders::Line;
                //         }
                //     },
                //     None => {}
                // }
            }
            else {
                gender = Genders::Female;
            }

                /*       
            println!("{} - direction: {:?} - RU: {:?} - RD: {:?} - gender: {:?} - diff: {:?} - {:?} - {:?} - {:?}", puzzle.file_name, direction,
            puzzle.right_up_corner,
            puzzle.right_down_corner,
            gender,
            distance_between_points(&puzzle.right_up_corner, &Point::new(convex.x, convex.y)),
            distance_between_points(&puzzle.right_up_corner, &Point::new(convex.x+ convex.width, convex.y)),
            distance_between_points(&puzzle.right_down_corner, &Point::new(convex.x, convex.y + convex.height)),
            distance_between_points(&puzzle.right_down_corner, &Point::new(convex.x + convex.width, convex.y + convex.height))
            );
            */
        },
        Direction::UpSide => {
            let max_corner;
            if puzzle.left_up_corner.y > puzzle.right_up_corner.y {
                max_corner = puzzle.right_up_corner.y;
            }
            else {
                max_corner = puzzle.left_up_corner.y;
            }

            if max_corner - puzzle.y_min.y > 100 {
                gender = Genders::Male;
            }
            else if convex.width < 200 || convex.height < 200 {
                gender = Genders::Line;
                // let max_value = vec![
                //     distance_between_points(&Point::new(convex.x, convex.y), &puzzle.left_up_corner)?,
                //     distance_between_points(&puzzle.left_up_corner, &Point::new(convex.x, convex.y + convex.height))?,
                //     distance_between_points(&puzzle.right_up_corner, &Point::new(convex.x + convex.width, convex.y))?,
                //     distance_between_points(&puzzle.right_up_corner, &Point::new(convex.x + convex.width, convex.y + convex.height))?
                // ];
                // dbg!(&max_value);

                //     match max_value.iter().max() {
                //     Some(val) => {
                //         if val > &100 {
                //             gender = Genders::Female;
                //         }
                //         else {
                //             gender = Genders::Line;
                //         }
                //     },
                //     None => {}
                // }
            }
            else {
                gender = Genders::Female;
            }
            /* 
            println!("{} - direction: {:?} - LU: {:?} - RU: {:?} - gender: {:?} - diff: {:?} - {:?} - {:?} - {:?}", puzzle.file_name, direction,
            puzzle.left_up_corner,
            puzzle.right_up_corner,
            gender,
            distance_between_points(&Point::new(convex.x, convex.y), &puzzle.left_up_corner),
            distance_between_points(&puzzle.left_up_corner, &Point::new(convex.x, convex.y + convex.height)),
            distance_between_points(&puzzle.right_up_corner, &Point::new(convex.x + convex.width, convex.y)),
            distance_between_points(&puzzle.right_up_corner, &Point::new(convex.x + convex.width, convex.y + convex.height))
            );
            */
        }
    }
    // println!("{} - direction: {:?} - gender: {:?}",&puzzle.file_name, direction, gender);
    Ok(gender)
}


fn split_contour(puzzle: &mut PuzzlePiece) -> Result<(VectorOfVectorOfPoint, Vec<ContourWithDir>), Error> {
    let mut contour_values = VectorOfVectorOfPoint::new();
    let mut contour_values_with_dir = Vec::new();

    for dir in Direction::iter() {
        let (single_contours, countour_traslated, x_max, y_min, y_max) = split_single_contour(puzzle, dir)?;
        let gender = get_gender(puzzle, dir, &single_contours)?;
        let mut c = ContourWithDir::new(single_contours.clone(), dir, gender, countour_traslated, x_max, y_min, y_max);
        c.dx();
        contour_values_with_dir.push(c);
        contour_values.push(single_contours);
    }
    
    Ok((contour_values, contour_values_with_dir))
}

fn search_best_threshold(grey_phase: &Mat) -> Result<i32, Error> {
    let mut threshold = 0;
    let mut min_len = usize::MAX;
    for threshold_value in 160..230 {
        let (contours_cv, _phase) = sub_process(grey_phase, threshold_value)?;

        let mut len = 0;
        for first in &contours_cv {
            len = first.len();
            break;
        }

        //println!("len: {} - min_len: {} - threshold: {} - threshold_value: {}", len, min_len, threshold, threshold_value);
        //println!("{};{}", threshold_value, len);

        //if  len > 4000 && len < min_len {
        if len < min_len {
            min_len = len;
            threshold = threshold_value;
        }
    }

    //println!("min_len: {} - threshold: {}", min_len, threshold);

    Ok(threshold)
}

fn blur(phase: &Mat) -> Result<Mat, anyhow::Error> {
    let ksize = cv::core::Size::new(15,15);
    let mut new_phase = cv::core::Mat::default();

    cv::imgproc::blur_def (
        &phase,
        &mut new_phase,
        ksize
    )?;
    Ok(new_phase)
}

fn sub_process(grey_phase: &Mat, threshold_value: i32) -> Result<(VectorOfVectorOfPoint, Mat), Error> {
    let phase = threshold(grey_phase, threshold_value)?;
    let phase = bitwise(&phase)?;
    let phase = morph(&phase)?;
    let contour_values = find_contour(&phase)?;

    //cv::highgui::imshow("sub_process", &phase);
    //cv::highgui::wait_key(500)?;

    Ok((contour_values, phase))
}

fn find_centroid(puzzle: &PuzzlePiece) -> Result<(i32, i32), Error> {
    let mut cx = 0.0;
    let mut cy = 0.0;

    for first in &puzzle.original_contours {
        match cv::imgproc::moments_def(&first) {
            Ok(moment) =>         {
                // println!("moment: {:?}", moment);
                // println!("X: {}, Y: {}", moment.m10/moment.m00, moment.m01/moment.m00);
                cx = moment.m10/moment.m00;
                cy = moment.m01/moment.m00;
            },
            Err(err) => println!("Error: {:?}", err),
        }
        break;
    }

    Ok((cx as i32, cy as i32))
}

fn midpoint(point1: &Point, point2: &Point) -> Point {
    Point::new((point1.x + point2.x)/2, (point1.y + point2.y)/2)
}

fn get_polygon(puzzle: &PuzzlePiece, delta: i32, direction: &Direction, iteration: i32) -> VectorOfPoint {
    let mut polygon = VectorOfPoint::new();
    //polygon.push(Point::new(puzzle.cx, puzzle.cy));
    
    let mid1_down =  midpoint(&puzzle.left_down_corner, &puzzle.right_down_corner);
    let mut mid2_down = puzzle.center;
    if iteration > 1 {
        for _ in 1..iteration {
            mid2_down = midpoint(&mid1_down, &mid2_down);
        }
    }

    let mid1_up =  midpoint(&puzzle.left_up_corner, &puzzle.right_up_corner);
    let mut mid2_up = puzzle.center;
    if iteration > 1 {
        for _ in 1..iteration {
            mid2_up = midpoint(&mid1_up, &mid2_up);
        }
    }

    match direction {
        Direction::DownSide => {
            polygon.push(puzzle.left_down_corner.clone());
            polygon.push(Point::new(puzzle.left_down_corner.x, 
                puzzle.left_down_corner.y + (puzzle.y_max.y - puzzle.left_down_corner.y) + delta));
            polygon.push(Point::new(puzzle.right_down_corner.x, 
                puzzle.right_down_corner.y + (puzzle.y_max.y - puzzle.right_down_corner.y) + delta));
            polygon.push(puzzle.right_down_corner.clone());
            polygon.push(mid2_down);
            //println!("Direction: {:?} - Point1: {:?} - Point2: {:?}", direction, puzzle.left_down_corner, puzzle.right_down_corner);
        },
        Direction::UpSide => {
            polygon.push(puzzle.left_up_corner.clone());
            polygon.push(Point::new(puzzle.left_up_corner.x, 
                puzzle.left_up_corner.y - (puzzle.left_up_corner.y - puzzle.y_min.y) - delta));
            polygon.push(Point::new(puzzle.right_up_corner.x, 
                puzzle.right_up_corner.y - (puzzle.right_up_corner.y - puzzle.y_min.y) - delta));
            polygon.push(puzzle.right_up_corner.clone());
            polygon.push(mid2_up);
            //println!("Direction: {:?} - Point1: {:?} - Point2: {:?}", direction, puzzle.left_up_corner, puzzle.right_up_corner);
        },
        Direction::RightSide => {
            polygon.push(puzzle.right_up_corner.clone());
            polygon.push(Point::new(puzzle.right_up_corner.x 
                + (puzzle.x_max.x - puzzle.right_up_corner.x) + delta, puzzle.right_up_corner.y));
            polygon.push(Point::new(puzzle.right_down_corner.x 
                + (puzzle.x_max.x - puzzle.right_down_corner.x) + delta, puzzle.right_down_corner.y));
            polygon.push(puzzle.right_down_corner.clone());
            // let mid1 =  (puzzle.right_up_corner + puzzle.right_down_corner) / 2;
            // let mid2 = (mid1 + center) / 2;
            polygon.push(mid2_down);
            polygon.push(mid2_up);
            //println!("Direction: {:?} - Point1: {:?} - Point2: {:?}", direction, puzzle.right_up_corner, puzzle.right_down_corner);
        },
        Direction::LeftSide => {
            polygon.push(puzzle.left_up_corner.clone());
            polygon.push(Point::new(puzzle.left_up_corner.x
                - (puzzle.left_up_corner.x - puzzle.x_min.x) - delta, puzzle.left_up_corner.y));
            polygon.push(Point::new(puzzle.left_down_corner.x
                - (puzzle.left_down_corner.x - puzzle.x_min.x) - delta, puzzle.left_down_corner.y));
            polygon.push(puzzle.left_down_corner.clone());
            // let mid1 =  (puzzle.left_up_corner + puzzle.left_down_corner) / 2;
            // let mid2 = (mid1 + center) / 2;
            polygon.push(mid2_down);
            polygon.push(mid2_up);
            //println!("Direction: {:?} - Point1: {:?} - Point2: {:?}", direction, puzzle.left_up_corner, puzzle.left_down_corner);
        },
    }

    polygon
}

fn get_color() -> cv::core::Scalar {
    let mut rng = rand::thread_rng();
    let n1 = rng.gen_range(0.0..255.0);
    let n2 = rng.gen_range(0.0..255.0);
    let n3 = rng.gen_range(0.0..255.0);
    cv::core::Scalar::new(n1, n2, n3, 255.0)
}

fn show_image(text: &str, img: &Mat) {
    let _ = cv::highgui::imshow(text, img);
}

fn find_bounding_rect(puzzle: &PuzzlePiece, _original_image: &Mat, contour: &VectorOfVectorOfPoint) -> Result<cv::core::Rect, anyhow::Error> {
    let mut rect = cv::core::Rect::default();

    let mut max_rect = cv::core::Rect::default();
    for first in contour {
        rect = match cv::imgproc::bounding_rect(&first) {
            Ok(val) => {
                val
            },
            Err(err) => {
                println!("find_bounding_rect - file_name: {} - err: {}", puzzle.file_name, err);
                return Err(anyhow!(err));
            }
        };
        if rect.width > max_rect.width {
            max_rect = rect;
        }
    }

    /*
    let mut phase = original_image.clone();
    let mut contours = VectorOfPoint::default();
    let point1 = Point::new(rect.x, rect.y);
    let point2 = Point::new(rect.x + rect.width, rect.y+rect.height);
    let thickness = 20;
    let line_type = cv::imgproc::LINE_8;
    let shift = 0;
    cv::imgproc::rectangle_points(
        &mut phase,
        point1,
        point2,
        get_color(),
        thickness,
        line_type,
        shift
    )?;

    let zero_offset = Point::new(0, 0);

    match cv::imgproc::draw_contours(&mut phase,
        &contour,
        -1,
        get_color(),
        thickness,
        cv::imgproc::LINE_8,
        &cv::core::no_array(),
        2,
        zero_offset){
            Ok(_) => {},
            Err(err) => {
                println!("Error on draw_contours - {}", err);
                return Err(anyhow!(err));
            }
        } 

    let _  = cv::highgui::imshow("find_bounding_rec", &phase);
    */
    Ok(rect)
}

fn split_single_contour(puzzle: &mut PuzzlePiece, direction: Direction) -> Result<(VectorOfPoint, VectorOfPoint, i32, i32, i32), Error> {
    let mut vector = VectorOfPoint::new();

    //print!("{} -> Start -> {:?} -> ", puzzle.file_name, direction);
    for first in &puzzle.original_contours {
        let mut onda;
        let mut count;
        for i in 0..10 {
            onda = 0;
            count = 0;
            vector.clear();
            let polygon = get_polygon(&puzzle, 100, &direction, i);
            for point in first.iter() {
                match cv::imgproc::point_polygon_test(
                    &polygon,
                    cv::core::Point2f::new(point.x as f32, point.y as f32),
                    true){
                        Ok(val) => {
                            if val > 0.0 {
                                if onda != 1 {
                                    onda = 1;
                                    count += 1;
                                    //print!("1");
                                }
                                //println!("{:?} -> x: {}, y: {} -> val: {}", direction, point.x, point.y, val);
                                vector.push(Point::new(point.x, point.y));  
                            }
                            else if onda != 2 {
                                onda = 2;
                                count += 1;
                                //print!("2")
                            }
                        },
                        Err(err) => {println!("Error on split_single_contour: {}", err);}
                    }
            }
            if count <= 3 {
                puzzle.polygon.insert(direction, polygon.clone());
                break;
            }
            else {
                //println!("{} - {:?} - Pezzo con problemi -> {}", puzzle.file_name, direction, i);
            }
        }         
    //print!("\n");
        break;
    }

    //println!("{} - {:?} - {:?}:{:?}", puzzle.file_name, direction, vector.get(0)?, vector.get(vector.len() - 1)?);

    let (up_left_point, _down_right_point) = get_extreme(direction, &vector)?;

    let mut vector_after_first_traslate = VectorOfPoint::new();
    for point in vector.iter() {
        vector_after_first_traslate.push(point - up_left_point);
    }

    let (up_left_point_translated, down_right_point_translated) = get_extreme(direction, &vector_after_first_traslate)?;
    let mut angle = (down_right_point_translated.y as f64/down_right_point_translated.x as f64).atan();

    //println!("{} - {:?} - {:?}:{:?} - angle: {}->{}", puzzle.file_name, direction, up_left_point_translated, down_right_point_translated, angle, angle.to_degrees());
    if angle < 0.0 {
        if down_right_point_translated.y < 0 {
            angle = (2.0 * std::f64::consts::PI + angle).abs() as f64;
        }
        else {
            angle = (std::f64::consts::PI + angle).abs() as f64;
        }
        //println!("{} - {:?} - {:?}:{:?} - angle: {}->{}", puzzle.file_name, direction, up_left_point_translated, down_right_point_translated, angle, angle.to_degrees());
    }

    let m = cv::imgproc::get_rotation_matrix_2d (
        cv::core::Point2f::new(up_left_point_translated.x as f32, up_left_point_translated.y as f32),
        angle.to_degrees(),
        1.0
    )?;

    let mut vector_rotated = vector_after_first_traslate.clone();

    cv::core::transform(
        &vector_after_first_traslate,
        &mut vector_rotated,
        &m
    )?;

    let mut y_max = 0;
    for point in vector_rotated.iter() {
        if y_max < point.y {
            y_max = point.y;
        }
    }

    let mut vector_traslated = VectorOfPoint::new();

    if y_max > 100 {
        for point in vector_rotated.iter() {
            vector_traslated.push(Point::new(point.x, -point.y));
        }
    }
    else {
        vector_traslated = vector_rotated.clone();
    }

    let mut x_max = 0;
    let mut y_min = i32::MAX;
    let mut y_max = 0;

    for point in vector_traslated.iter() {
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

    /* if y_min.abs() > 100 || y_max.abs() > 100 {
        println!("{} - {:?} - {}, {}, {} - {}",puzzle.file_name, direction, x_max, y_min, y_max, y_max - y_min);
    } */

    Ok((vector, vector_traslated, x_max, y_min, y_max))
}

fn get_extreme(direction: Direction, vector: &VectorOfPoint) -> Result<(Point, Point), Error> {
    let mut up_left_point = Point::new(0,0);
    let mut down_right = Point::new(0,0);
    match direction {
        Direction::DownSide | Direction::UpSide => {
            let mut x_min = i32::MAX;
            let mut x_max = 0;

            for index in 0..vector.len() {
                //println!("{:?}", vector.get(i)?);
                let x = vector.get(index)?.x;
                if x < x_min {
                    x_min = x;
                    up_left_point = vector.get(index)?;
                }

                if x > x_max {
                    x_max = x;
                    down_right = vector.get(index)?;
                }
            }
        },
        Direction::LeftSide | Direction::RightSide => {
            let mut y_min = i32::MAX;
            let mut y_max = 0;

            for index in 0..vector.len() {
                //println!("{:?}", vector.get(i)?);
                let y = vector.get(index)?.y;
                if y < y_min {
                    y_min = y;
                    up_left_point = vector.get(index)?;
                }

                if y > y_max {
                    y_max = y;
                    down_right = vector.get(index)?;
                }
            }
        },
    }
    Ok((up_left_point, down_right))
}

fn find_corners(_puzzle: &PuzzlePiece, phase: &Mat) -> Result<(f64, VectorOfPoint), anyhow::Error> {

    let mut corners = VectorOfPoint::new();
    let max_corners = 4;
    let quality_level = 0.1;
    let mut distance = 1100.0;
    let mut block_size;
    let use_harris_detector: bool = true;
    let k: f64 = 0.1;

    let mut min_corners = VectorOfPoint::new();
    // let mut min_norm = 0.0;
    let mut points = VectorOfPoint::default();
    let contour_center = Point::new(_puzzle.cx, _puzzle.cy); 
    // let mut min_distance = 0.0;

    // let mut max_point1 = 0.0;
    // let mut max_point2 = 0.0;
    // let mut max_point3 = 0.0;
    // let mut max_point4 = 0.0;
    let mut max_tot_distance = 0.0;
    loop {
        block_size = 80;
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
                k
            ){
                Ok(_) => {},
                Err(err) => println!("Error on find_corners (block_size {}): {} with ", block_size, err)
            };

            if corners.len() == 4 {
                // dbg!(distance, block_size);
                let value_min_area = cv::imgproc::min_area_rect(&corners)?;

                // let norm = cv::core::norm_def(&corners)?;
                
                // dbg!(&norm);

                points.clear();            
                let min_area_center = Point::new(value_min_area.center.x as i32, value_min_area.center.y as i32);
                let diff = min_area_center - contour_center;
                points.push(diff);
                // let distance_norm = cv::core::norm_def(&points)?;

                let p1_diff = corners.get(0)? - contour_center;
                points.clear();
                points.push(p1_diff);
                // let distance_p1 = cv::core::norm_def(&points)?;

                let p2_diff = corners.get(1)? - contour_center;
                points.clear();
                points.push(p2_diff);
                // let distance_p2 = cv::core::norm_def(&points)?;

                let p3_diff = corners.get(2)? - contour_center;
                points.clear();
                points.push(p3_diff);
                // let distance_p3 = cv::core::norm_def(&points)?;

                let p4_diff = corners.get(3)? - contour_center;
                points.clear();
                points.push(p4_diff);
                // let distance_p4 = cv::core::norm_def(&points)?;

                points.push(p1_diff);
                points.push(p2_diff);
                points.push(p3_diff);
                let tot_distance = cv::core::norm_def(&points)?;

                // if max_point1 <= distance_p1 && max_point2 <= distance_p2 && max_point3 <= distance_p3 && max_point4 <= distance_p4 {
                //     max_point1 = distance_p1;
                //     max_point2 = distance_p2;
                //     max_point3 = distance_p3;
                //     max_point4 = distance_p4;
                //     min_corners = corners.clone();
                //     // println!("Presa")
                // }

                if max_tot_distance < tot_distance {
                    min_corners = corners.clone();
                    max_tot_distance = tot_distance;
                    // println!("{};{};{};{};{};{};{}", norm, distance_norm, distance_p1, distance_p2, distance_p3, distance_p4, tot_distance);
                    //println!("Presa");
                }

                //println!("{};{};{};{};{};{};{}", norm, distance_norm, distance_p1, distance_p2, distance_p3, distance_p4, tot_distance);

                // if norm > min_norm || min_distance < distance_norm {
                //     min_norm = norm;
                //     min_distance = distance_norm;
                //     min_corners = corners.clone();
                //     // dbg!(&min_norm, distance, block_size);
                // }
            }
            

            // dbg!(distance_norm);
            

            block_size += 5;
            if block_size > 110 {
                break;
            }
        }
        distance += 50.0;
        if distance > 1300.0 {
            break;
        }
    }

    //println!("cx: {:?} - cy: {:?}", _puzzle.cx, _puzzle.cy);

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

fn find_min_max(puzzle: &mut PuzzlePiece) -> std::io::Result<()> {
    for first in &puzzle.original_contours {
        for point in first.iter() {
            if point.x < puzzle.x_min.x {
                puzzle.x_min.x = point.x;
                puzzle.x_min.y = point.y;
            }

            if point.x > puzzle.x_max.x {
                puzzle.x_max.x = point.x;
                puzzle.x_max.y = point.y;
            }

            if point.y < puzzle.y_min.y {
                puzzle.y_min.x = point.x;
                puzzle.y_min.y = point.y;
            }

            if point.y > puzzle.y_max.y {
                puzzle.y_max.x = point.x;
                puzzle.y_max.y = point.y;
            }
        }
    }

    Ok(())
}

fn to_grey(phase: &Mat) -> Result<Mat, anyhow::Error> {
    let mut new_phase = cv::core::Mat::default();
    match cv::imgproc::cvt_color(&phase, 
        &mut new_phase, 
        cv::imgproc::COLOR_BGR2GRAY, 
        0){
            Ok(_) => {},
            Err(err) => {
                println!("To Grey Error: {}", err);
                return Err(anyhow!(err));
            }
        }

    Ok(new_phase)
}

fn draw_contour(puzzle: &PuzzlePiece) -> Result<(), anyhow::Error> {

    let mut phase = puzzle.original_image.clone();
    let zero_offset = Point::new(0, 0);
    let thickness: i32 = 20;

    // let rng = rand::thread_rng();

    // Disegna i 4 contorni con colori diversi
    /* for index in 0..puzzle.contours.len() {
        match cv::imgproc::draw_contours(&mut phase,
            &puzzle.contours,
            index as i32,
            get_color(),
            thickness,
            cv::imgproc::LINE_8,
            &cv::core::no_array(),
            2,
            zero_offset){
                Ok(_) => {},
                Err(err) => {
                    println!("Error on draw_contours - file_name: {} - index {}: {}", puzzle.file_name, index, err);
                    return Err(anyhow!(err));
                }
            }
    } */

    let mut countours = VectorOfVectorOfPoint::new();

    for dir in puzzle.contours_with_dir.iter() {
        countours.push(dir.countour.clone());
    }
    for index in 0..puzzle.contours.len() {
        match cv::imgproc::draw_contours(&mut phase,
            &countours,
            index as i32,
            get_color(),
            thickness,
            cv::imgproc::LINE_8,
            &cv::core::no_array(),
            2,
            zero_offset){
                Ok(_) => {},
                Err(err) => {
                    println!("Error on draw_contours - file_name: {} - index: {} {}", puzzle.file_name, index, err);
                    return Err(anyhow!(err));
                }
            }
    }

    let mut countours = VectorOfVectorOfPoint::new();

    for dir in puzzle.contours_with_dir.iter() {
        countours.push(dir.countour_traslated.clone());
    }
    for index in 0..puzzle.contours.len() {
        match cv::imgproc::draw_contours(&mut phase,
            &countours,
            index as i32,
            get_color(),
            thickness,
            cv::imgproc::LINE_8,
            &cv::core::no_array(),
            2,
            zero_offset){
                Ok(_) => {},
                Err(err) => {
                    println!("Error on draw_contours - file_name: {} - index: {} {}", puzzle.file_name, index, err);
                    return Err(anyhow!(err));
                }
            }
    }

    // Disegna l'area con cui ha calcolato se il contorno e dentro il poligono oppure no
    for dir in Direction::iter() {        
        if let Some(polygon) = puzzle.polygon.get(&dir) {
            cv::imgproc::polylines(
                &mut phase,
                &polygon,
                true,
                get_color(),
                10,
                cv::imgproc::LINE_8,
                0
            )?;
        }
    }

    // Disegna i 4 angoli
    for point in puzzle.corners.iter() {
        cv::imgproc::circle (
            &mut phase,
            point,
            20,
            cv::core::Scalar::new(0.0, 0.0, 255.0, 255.0),
            cv::imgproc::FILLED,
            cv::imgproc::LINE_8,
            0
        )?;
    }

    let name = format!("./{}_contours.jpg", puzzle.file_name);
    cv::imgcodecs::imwrite(&name, &phase, &cv::core::Vector::default())?;

    Ok(())
}

fn draw_simple_contour(puzzle: &PuzzlePiece) -> Result<(), anyhow::Error> {

    let mut phase = puzzle.original_image.clone();
    let zero_offset = Point::new(0, 0);
    let thickness: i32 = 20;

    let mut rng = rand::thread_rng();

    for index in 0..puzzle.original_contours.len() {
        let n1 = rng.gen_range(0.0..255.0);
        let n2 = rng.gen_range(0.0..255.0);
        let n3 = rng.gen_range(0.0..255.0);
        let color = cv::core::Scalar::new(n1, n2, n3, 255.0);
        match cv::imgproc::draw_contours(&mut phase,
            &puzzle.original_contours,
            index as i32,
            color,
            thickness,
            cv::imgproc::LINE_8,
            &cv::core::no_array(),
            2,
            zero_offset){
                Ok(_) => {},
                Err(err) => {
                    println!("Error on draw_contours - index {}: {}", index, err);
                    return Err(anyhow!(err));
                }
            }
            break;
    }

    // Disegna gli angoli
    let color = cv::core::Scalar::new(0.0, 0.0, 255.0, 255.0);
    for point in puzzle.corners.iter() {
        cv::imgproc::circle (
            &mut phase,
            point,
            20,
            color,
            cv::imgproc::FILLED,
            cv::imgproc::LINE_8,
            0
        )?;
    }

    // Disegna il centro
    let point = Point::new(puzzle.cx, puzzle.cy);
    cv::imgproc::circle (
        &mut phase,
        point,
        20,
        color,
        cv::imgproc::FILLED,
        cv::imgproc::LINE_8,
        0
    )?;
    
    let _name = format!("./{}_simple_contours.jpg", puzzle.file_name);
    //cv::imgcodecs::imwrite(&name, &phase, &cv::core::Vector::default())?;

    Ok(())
}

fn find_contour(phase: &Mat) -> Result<VectorOfVectorOfPoint, anyhow::Error> {
    let mut original_contour_values = VectorOfVectorOfPoint::new();
    let mut contour_values = VectorOfVectorOfPoint::new();
    cv::imgproc::find_contours(
        &phase, 
        &mut original_contour_values, 
        cv::imgproc:: RETR_TREE,
        cv::imgproc::CHAIN_APPROX_SIMPLE,
        Point::new(0, 0),
    )?;

    let mut biggest = 0;
    // Take only the first contour with len greater than 1000
    for first in &original_contour_values {
        if biggest < first.len() {
            biggest = first.len();
        }
    }

    for first in &original_contour_values {
        if biggest == first.len() {
            contour_values.push(first);
            break;
        }
    }

    Ok(contour_values)
}

fn morph(phase: &Mat) -> Result<Mat, anyhow::Error> {
    let mut new_phase = cv::core::Mat::default();
    let anchor = Point::new(-1, -1);
    let ksize = cv::core::Size::new(5,5);
    let kernel = cv::imgproc::get_structuring_element(
        0,
        ksize,
        anchor
    )?;
    cv::imgproc::morphology_ex(
        &phase,
        &mut new_phase,
        cv::imgproc::MORPH_OPEN,
        &kernel,
        anchor,
        1,
        cv::core::BORDER_CONSTANT,
        cv::imgproc::morphology_default_border_value()?)?;

    Ok(new_phase)
}

fn bitwise(phase: &Mat) -> Result<Mat, anyhow::Error> {
    let mut new_phase = cv::core::Mat::default();
    cv::core::bitwise_not_def(
        &phase,
        &mut new_phase
    )?;

    Ok(new_phase)
}

fn threshold(phase: &Mat, threshold_value: i32) -> Result<Mat, anyhow::Error> {
    let mut new_phase = cv::core::Mat::default();
    cv::imgproc::threshold(
        &phase,
        &mut new_phase,
        threshold_value as f64,
        255.0,
        cv::imgproc::THRESH_BINARY
    )?;

    Ok(new_phase)
}

fn max_x_for_y(contour: &VectorOfPoint, x_y_max_sequence: i32) -> Result<i32>
{
    let mut max_sporgenza = 0;
    let len = contour.len();
    for j in x_y_max_sequence..-100 {
        let mut v = Vec::new();
        for i in 0..len {
            // Quanti X hanno questa altezza?
            let point = contour.get(i)?;
            if point.y == j {
                v.push(point.x);
            }
        }
        if v.len() > 1 {
            if v[0] > v[v.len() - 1] {
                let val = v[0] - v[v.len() - 1];
                if val > max_sporgenza {
                    max_sporgenza = val;
                }
            }
            else {
                let val = v[v.len() - 1] - v[0];
                if val > max_sporgenza {
                    max_sporgenza = val;
                }
            }
        }
    }

    Ok(max_sporgenza)
}

fn match_shapes(puzzle1: &PuzzlePiece, puzzle2: &PuzzlePiece) -> Result<(), anyhow::Error>{
    //fill_only_contour(puzzle1, &puzzle2)?;
    //println!("{}-{}", puzzle1.file_name, puzzle2.file_name);
    //println!("");
    for sequence1 in &puzzle1.contours_with_dir {

/*         let mut hu_moments_seq_1 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        if sequence1.gender != Genders::Line {
            hu_moments_seq_1 = match cv::imgproc::moments_def(&sequence1.countour) {
                Ok(moment) => {
                    let mut hu_moments:[f64;7] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
                    // println!("moment: {:?}", moment);
                    // println!("X: {}, Y: {}", moment.m10/moment.m00, moment.m01/moment.m00);
                    cv::imgproc::hu_moments(moment, &mut hu_moments)?;
                    hu_moments
                },
                Err(err) => [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            };
        } */

        for sequence2 in &puzzle2.contours_with_dir {
            if sequence1.gender != Genders::Line && sequence2.gender != Genders::Line && sequence1.gender != sequence2.gender {
                //sequence1.dx();
                //sequence2.dx();
                let d1 = (sequence1.d1 - sequence2.d1).abs();
                let d2 = (sequence1.d2 - sequence2.d2).abs();
                let d3 = (sequence1.d3 - sequence2.d3).abs();
                let d4 = (sequence1.d4 - sequence2.d4).abs();
                let d5 = (sequence1.d5 - sequence2.d5).abs();
                let d1_d4 = (sequence1.d1 - sequence2.d4).abs();
                let d4_d1 = (sequence1.d4 - sequence2.d1).abs();
                if d5 < 100 && d3 < 30 && 
                    ((d1 < 60 && d4 < 60) || (d1_d4 < 60 && d4_d1 < 60))
                {
                    println!("{} - {} - {:?} - {:?}; d1: {}; d2: {}; d3: {}, d4: {}; d5: {}; d1-d4: {}; d4-d1: {}",
                        puzzle1.file_name, puzzle2.file_name, sequence1.dir, sequence2.dir,
                        d1,
                        d2,
                        d3,
                        d4,
                        d5,
                        d1_d4,
                        d4_d1,
                    )
                };
            /* 
                let len1 = sequence1.countour_traslated.len();
                let len2 = sequence2.countour_traslated.len();
                let min_len = if  len1 > len2 { len2 } else { len1 };
                let diff_lunghezza = if  len1 > len2 { len1 - len2 } else { len2 - len1 };
                let diff_larghezza = (sequence1.x_max - sequence2.x_max).abs();
                let diff_altezza = ((sequence1.y_min - sequence1.y_max).abs() - (sequence2.y_min - sequence2.y_max).abs()).abs();
                let prod1 = if  len1 > len2 { 1 } else { 0 };
                let prod2 = if  len1 > len2 { 0 } else { 1 };

                // Calcola la X in cui la Y è massima
                let mut x_y_max_sequence_1 = 0;
                let mut y_max_1 = 0;
                for point in sequence1.countour_traslated.iter() {
                    if point.y < y_max_1 {
                        y_max_1 = point.y;
                        x_y_max_sequence_1 = point.x;
                    }
                }

                // Calcola la X in cui la Y è massima
                let mut x_y_max_sequence_2 = 0;
                let mut y_max_2 = 0;
                for point in sequence2.countour_traslated.iter() {
                    if point.y < y_max_2 {
                        y_max_2 = point.y;
                        x_y_max_sequence_2 = point.x;
                    }
                }

                // Fa la differenza delle Y per capire quanto le figure siano simili sequence1 - sequence2
                // Prova anche a spostare una figura rispetto all'altra per cercare la minima differenza
                let mut min_count_0 = i32::MAX;
                for j in 0..diff_lunghezza {
                    let mut count = 0;
                    for i in 0..min_len {
                        count += sequence1.countour_traslated.get(i + j*prod1)?.y - sequence2.countour_traslated.get(i + j*prod2)?.y;
                    }

                    if min_count_0 > count.abs() {
                        min_count_0 = count.abs();
                    }
                }

                // Fa la differenza delle Y per capire quanto le figure siano simili sequence2 - sequence1
                // Prova anche a spostare una figura rispetto all'altra per cercare la minima differenza
                let mut min_count_3 = i32::MAX;
                for j in 0..diff_lunghezza {
                    let mut count = 0;
                    for i in 0..min_len {
                        count += sequence2.countour_traslated.get(i + j*prod2)?.y - sequence1.countour_traslated.get(i + j*prod1)?.y;
                    }

                    if min_count_3 > count.abs() {
                        min_count_3 = count.abs();
                    }
                }

                // Fa la differenza delle Y per la sequecen1 partendo da dx vs sx mentre per sequence2 da sx vs dx
                // Prova anche a spostare una figura rispetto all'altra per cercare la minima differenza
                let mut min_count_1 = i32::MAX;
                let len1 = len1 - 1;
                for j in 0..diff_lunghezza {
                    let mut count = 0;
                    let j_prod1 = j*prod1;
                    let j_prod2 = j*prod2;
                    for i in 0..min_len {
                        count += sequence1.countour_traslated.get(len1 - i - j_prod1)?.y - sequence2.countour_traslated.get(i + j_prod2)?.y;
                    }

                    if min_count_1 > count.abs() {
                        min_count_1 = count.abs();
                    }
                }

                // Fa la differenza delle Y per la sequecen2 partendo da dx vs sx mentre per sequence1 da sx vs dx
                // Prova anche a spostare una figura rispetto all'altra per cercare la minima differenza
                let mut min_count_2 = i32::MAX;
                let len2 = len2 - 1;
                for j in 0..diff_lunghezza {
                    let mut count = 0;
                    let j_prod1 = j*prod1;
                    let j_prod2 = j*prod2;
                    for i in 0..min_len {
                        count += sequence1.countour_traslated.get(i + j_prod1)?.y - sequence2.countour_traslated.get(len2 - i - j_prod2)?.y;
                    }

                    if min_count_2 > count.abs() {
                        min_count_2 = count.abs();
                    }
                }

                // Cerca la X massima della sporgenza nel sequence1
                let max_sporgenza_1 = max_x_for_y(&sequence1.countour_traslated, y_max_1)?;

                // Cerca la X massima della sporgenza nel sequence2
                let max_sporgenza_2 = max_x_for_y(&sequence2.countour_traslated, y_max_2)?;

                if diff_larghezza < 100 && diff_altezza < 30{
                    println!("{} - {} - {:?} - {:?}; count_0: {}; count_1: {}; count_2: {}; count_3: {}; diff_lunghezza: {}; diff_larghezza: {}; diff_altezza: {}; 
                    x_y_max_sequence: {}; ; x_y_max_sequence_1_rev: {}; x_y_max_sequence_2_rev: {};
                    max_sporgenza: {}", 
                        puzzle1.file_name, puzzle2.file_name, sequence1.dir, sequence2.dir, 
                        min_count_0, min_count_1, min_count_2, min_count_3,
                        diff_lunghezza,
                        diff_larghezza,
                        diff_altezza,
                        (x_y_max_sequence_1 - x_y_max_sequence_2).abs(),
                        (x_y_max_sequence_2 - (sequence1.countour_traslated.len() - x_y_max_sequence_1 as usize) as i32).abs(),
                        (x_y_max_sequence_1 - (sequence2.countour_traslated.len() - x_y_max_sequence_2 as usize) as i32).abs(),
                        (max_sporgenza_1 - max_sporgenza_2).abs()
                    );
                }

                */

                /* else {
                    println!("Eliminated - {} - {} - {:?} - {:?}; count: {}; diff_lunghezza: {}; diff_larghezza: {}; diff_altezza: {}", 
                        puzzle1.file_name, puzzle2.file_name, sequence1.dir, sequence2.dir, count.abs(), diff_lunghezza,
                        diff_larghezza,
                        diff_altezza
                    );
                } */
/* 
                let mut angle = 45.0;
                let scale = 1.0;
                let mut best_rect = Mat::default();
                let mut width = i32::MAX;
                let mut cx = 0.0;
                let mut cy = 0.0;            

                match cv::imgproc::moments_def(&sequence1.countour) {
                    Ok(moment) =>         {
                        // println!("moment: {:?}", moment);
                        // println!("X: {}, Y: {}", moment.m10/moment.m00, moment.m01/moment.m00);
                        cx = moment.m10/moment.m00;
                        cy = moment.m01/moment.m00;
                    },
                    Err(err) => println!("Error: {:?}", err),
                }

                let m1 = cv::imgproc::get_rotation_matrix_2d (
                    cv::core::Point2f::new(cx as f32, cy as f32),
                    angle,
                    scale
                )?;

                let mut my_contour = VectorOfPoint::new();
                for point in &sequence1.countour {
                    my_contour.push(point);
                }
                cv::core::transform(
                    &sequence1.countour,
                    &mut my_contour,
                    &m1
                )?;
            */
            

            /* let hu_moments_seq_2 = match cv::imgproc::moments_def(&sequence2.countour) {
                Ok(moment) => {
                    let mut hu_moments:[f64;7] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
                    // println!("moment: {:?}", moment);
                    // println!("X: {}, Y: {}", moment.m10/moment.m00, moment.m01/moment.m00);
                    cv::imgproc::hu_moments(moment, &mut hu_moments)?;
                    hu_moments
                },
                Err(err) => [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            };

            println!("{:?} | {:.6} | {:.6} | {:.6} | {:.6} | {:.6} | {:.6} | {:.6}", sequence1.dir,
            hu_moments_seq_1[0],
            hu_moments_seq_1[1],
            hu_moments_seq_1[2],
            hu_moments_seq_1[3],
            hu_moments_seq_1[4],
            hu_moments_seq_1[5],
            hu_moments_seq_1[6]
            );
            println!("{:?} | {:.6} | {:.6} | {:.6} | {:.6} | {:.6} | {:.6} | {:.6}", sequence2.dir,
            hu_moments_seq_2[0],
            hu_moments_seq_2[1],
            hu_moments_seq_2[2],
            hu_moments_seq_2[3],
            hu_moments_seq_2[4],
            hu_moments_seq_2[5],
            hu_moments_seq_2[6]
            );
            println!(""); */

                /* print!("{:?}-{:?}", sequence1.dir, sequence2.dir);
                for method in 1..4 {
                    let m = cv::imgproc::match_shapes(
                        &sequence1.countour,
                        &sequence2.countour,
                        method,
                        1.0
                    )?;
                    //fill_only_contour_with_text(&puzzle1, &sequence1, &sequence2, format!("{:?}-{:?}: {}", sequence1.dir, sequence2.dir,m).to_string())?;
                    if m > 0.0 {
                        print!(" - {:?}", m);
                    }
                }
                println!(""); */
            }
        }
    }

    Ok(())
}

fn fill_only_contour(puzzle: &PuzzlePiece, side1: &ContourWithDir, side2: &ContourWithDir)-> Result<(), anyhow::Error> {
    let mut new_phase = cv::core::Mat::new_size_with_default(
        puzzle.original_image.size()?,
        cv::core::CV_32FC1,
        get_black_color(), 
    )?;

    match cv::imgproc::fill_poly_def(
        &mut new_phase,
        &side1.countour,
        get_color(),
    ){
        Ok(_) => {},
        Err(err) => println!("Error on fill_convex_poly: {}", err)
    }

    match cv::imgproc::fill_poly_def(
        &mut new_phase,
        &side2.countour,
        get_color(),
    ){
        Ok(_) => {},
        Err(err) => println!("Error on fill_convex_poly: {}", err)
    }
    //show_image("fill_only_contour", &new_phase);
    //wait_key(0);

    Ok(())
}

fn fill_only_contour_with_text(puzzle: &PuzzlePiece, side1: &ContourWithDir, side2: &ContourWithDir, text: String)-> Result<(), anyhow::Error> {
    let mut new_phase = cv::core::Mat::new_size_with_default(
        puzzle.original_image.size()?,
        cv::core::CV_32FC1,
        get_black_color(), 
    )?;

    match cv::imgproc::fill_poly_def(
        &mut new_phase,
        &side1.countour,
        get_color(),
    ){
        Ok(_) => {},
        Err(err) => println!("Error on fill_convex_poly: {}", err)
    }

    match cv::imgproc::fill_poly_def(
        &mut new_phase,
        &side2.countour,
        get_color(),
    ){
        Ok(_) => {},
        Err(err) => println!("Error on fill_convex_poly: {}", err)
    }

    cv::imgproc::put_text_def(
        &mut new_phase,
        &text,
        Point::new(100, 300),
        3,
        3.0,
        get_white_color()
    )?;

    show_image("fill_only_contour", &new_phase);
    let _ = wait_key(0);

    Ok(())
}