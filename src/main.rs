use anyhow::Error;
use anyhow::anyhow;
use anyhow::Result;
use opencv::{self as cv, prelude::*};
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

#[derive(EnumIter, Debug, Hash, Eq, PartialEq, Clone)]
enum Direction {
    UpSide,
    DownSide,
    RightSide,
    LeftSide
}

struct PuzzlePiece {
    file_name: String,
    contours: VectorOfVectorOfPoint,

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
}

impl PuzzlePiece {
    fn new() -> Self {
        Self {
            file_name: "puzzle".to_string(),
            contours: VectorOfVectorOfPoint::default(),

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

    for puzzle1 in puzzles.iter() {
        for puzzle2 in puzzles.iter() {
            match_shapes(puzzle1, puzzle2)?;
        }
    }

    Ok(())
}

fn process(file_name: &str) -> Result<PuzzlePiece, anyhow::Error> {
    println!("Process: {} ...",file_name);
    let mut puzzle = PuzzlePiece::new();
    puzzle.file_name = file_name.to_string();
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

/* 
    let rotate = find_best_rotate(&puzzle)?;
    puzzle.original_image = rotate_image(&puzzle, &puzzle.original_image, &rotate)?;
    //show_image("Dopo", &puzzle.original_image);

    //wait_key(0);


    let phase = to_grey(&puzzle.original_image)?;
    let grey_phase = blur(&phase)?;
    puzzle.grey = grey_phase;

    let threshold = search_best_threshold(&puzzle.grey)?;
    let (contours, phase) = sub_process(&puzzle.grey, threshold)?;
    puzzle.original_contours = contours;
    puzzle.threshold = threshold;

    puzzle.rect = find_bounding_rect(&puzzle, &phase, &puzzle.original_contours)?;

    let Ok((cx, cy)) = find_centroid(&puzzle) else { todo!() };
    puzzle.cx = cx;
    puzzle.cy = cy;

    let _ = find_min_max(&mut puzzle);
*/
    
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


    //let corners = find_corners(&puzzle, &pre_corner)?;
    //set_corners(&mut puzzle, &corners);

    puzzle.corners = corners;


    let _ = draw_simple_contour(&puzzle);

    puzzle.contours = split_contour(&mut puzzle)?;

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

fn split_contour(puzzle: &mut PuzzlePiece) -> Result<VectorOfVectorOfPoint, Error> {
    let mut contour_values = VectorOfVectorOfPoint::new();

    for dir in Direction::iter() {
        let single_contours = split_single_contour(puzzle, dir)?;
        contour_values.push(single_contours);
    }
    
    Ok(contour_values)
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

fn split_single_contour(puzzle: &mut PuzzlePiece, direction: Direction) -> std::io::Result<VectorOfPoint> {
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

    Ok(vector)
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
    for first in &puzzle.contours {
        for point in first.iter() {
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
    for index in 0..puzzle.contours.len() {
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

fn match_shapes(puzzle1: &PuzzlePiece, puzzle2: &PuzzlePiece) -> Result<(), anyhow::Error>{
    let method = 0;

    let contour1 = VectorOfPoint::new();
    let contour2 = VectorOfPoint::new();

    let m = cv::imgproc::match_shapes(
        &contour1,
        &contour2,
        method,
        1.0
    )?;
    dbg!(m);

    Ok(())
}