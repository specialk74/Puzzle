use anyhow::anyhow;
use itertools::Itertools;
use opencv::core::Point;
use opencv::core::Vector;
use opencv::{self as cv, prelude::*};
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use serde_json::{from_str, to_writer_pretty};
use std::fs::File;
use std::path::Path;
use strum::IntoEnumIterator;

mod contour_with_dir;
mod cv_utils;
mod draw;
mod puzzle_piece;
mod single_contour_params;
mod utils;

use crate::contour_with_dir::*;
use crate::cv_utils::*;
use crate::puzzle_piece::*;
use crate::single_contour_params::SingleContourParams;
use utils::*;

fn my_contour() -> Result<(), anyhow::Error> {
    let mut puzzles: Vec<PuzzlePiece> = find_files("./input/")
        .into_par_iter()
        .map(|file_name| process(&file_name).unwrap_or_default())
        .collect();

    let mut puzzles_links = Vec::new();

    for (element1, element2) in puzzles.iter().tuple_combinations() {
        let (_, link1, link2) = match_shapes(element1, element2)?;
        puzzles_links.push(link1);
        puzzles_links.push(link2);
    }

    for puzzle in puzzles.iter_mut() {
        for puzzle_link in puzzles_links.iter_mut() {
            if puzzle.get_file_name() == puzzle_link.get_file_name() {
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

        if puzzle.ok && puzzle.get_write_json() {
            let _ = to_writer_pretty(
                &File::create(format!("./output/{}.json", puzzle.get_output_file()))?,
                &puzzle.contours_with_dir,
            );
        }
    }

    Ok(())
}

fn process(file_name: &str) -> Result<PuzzlePiece, anyhow::Error> {
    println!("Process: {} ...", file_name);
    let mut puzzle = PuzzlePiece::new(file_name);

    // region: Read json file
    // Check if json file exists. In case yes, load it.
    let output_file = format!("./output/{}.json", puzzle.get_output_file());
    if Path::new(&output_file).exists() {
        let val = std::fs::read_to_string(output_file)?;
        let mut u: Vec<ContourWithDir> = from_str(&val)?;
        for contour_with_dir in u.iter_mut() {
            contour_with_dir.clear_links();
        }
        puzzle.contours_with_dir = u;
        return Ok(puzzle);
    }
    // endregion

    puzzle.read_image()?;

    let phase = to_grey(&puzzle.original_image)?;
    puzzle.grey = blur(&phase)?;
    puzzle.search_best_threshold()?;
    puzzle.original_contours = sub_process(&puzzle.grey, puzzle.threshold)?;
    puzzle.center = find_centroid(&puzzle.original_contours)?;
    puzzle.find_min_max();
    let phase = fill_poly(&puzzle)?;

    let corners;
    let (max1, corners1) = find_corners(&puzzle.center, &puzzle.grey)?;
    let (max2, corners2) = find_corners(&puzzle.center, &phase)?;

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

    (puzzle.contours, puzzle.contours_with_dir) = split_contour(&mut puzzle)?;

    //println!("Draw contour: {}", &puzzle.file_name);
    draw_contour(&puzzle)?;

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

    let _name = format!("./output/{}_fill_convex_poly.jpg", puzzle.get_output_file());
    //cv::imgcodecs::imwrite(&name, &new_phase, &cv::core::Vector::default())?;

    Ok(new_phase)
}

fn get_gender(
    puzzle: &PuzzlePiece,
    direction: Direction,
    contour: &Vector<Point>,
) -> Result<Genders, anyhow::Error> {
    let convex = cv::imgproc::bounding_rect(contour)?;
    let small = convex.width < 200 || convex.height < 200;

    let gender = match direction {
        Direction::Down => {
            let max_corner = puzzle.left_down_corner.y.max(puzzle.right_down_corner.y);
            if puzzle.y_max.y - max_corner > 100 { Genders::Male }
            else if small { Genders::Line }
            else { Genders::Female }
        }
        Direction::Left => {
            let min_corner = puzzle.left_down_corner.x.min(puzzle.left_up_corner.x);
            if min_corner - puzzle.x_min.x > 100 { Genders::Male }
            else if small { Genders::Line }
            else { Genders::Female }
        }
        Direction::Right => {
            let max_corner = puzzle.right_up_corner.x.max(puzzle.right_down_corner.x);
            if puzzle.x_max.x - max_corner > 100 { Genders::Male }
            else if small { Genders::Line }
            else { Genders::Female }
        }
        Direction::Up => {
            let min_corner = puzzle.left_up_corner.y.min(puzzle.right_up_corner.y);
            if min_corner - puzzle.y_min.y > 100 { Genders::Male }
            else if small { Genders::Line }
            else { Genders::Female }
        }
    };
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
        //println!("Split single contour: {} - {:?}", puzzle.file_name, dir);
        let SingleContourParams {
            single_contours,
            countour_traslated,
            x_max,
            x_min: _x_min,
            y_min,
            y_max,
        } = match split_contour_by_direction(puzzle, dir) {
            Ok(params) => params,
            Err(err) => {
                println!("Err split_contour -> split_single_contour {:?}", err);
                return Err(anyhow!(err));
            }
        };
        //println!("Get Gender: {} - {:?}", puzzle.file_name, dir);
        let gender = match get_gender(puzzle, dir, &single_contours) {
            Ok(g) => g,
            Err(err) => {
                println!("Err split_contour -> get_gender: {:?}", err);
                return Err(anyhow!(err));
            }
        };
        //println!("Create ContourWithDir: {} - {:?}", puzzle.file_name, dir);
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

fn split_contour_by_direction(
    puzzle: &mut PuzzlePiece,
    direction: Direction,
) -> Result<SingleContourParams, anyhow::Error> {
    let mut single_contours = Vector::new();

    // println!(
    //     "Recupero il primo contour per la direzione: {:?} del file {:?}",
    //     direction, &puzzle.file_name
    // );
    let first = puzzle.first_contour()?;
    let mut onda;
    let mut count;
    for iteration in 0..100 {
        onda = 0;
        count = 0;
        single_contours.clear();
        // println!(
        //     "Iterazione #{} nella direzione {:?} per il file {:?}",
        //     iteration, direction, &puzzle.file_name
        // );
        // Crea il poligono che parte dal centro dell'immagine e arriva
        // alle 2 estremità nella direzione di direction;
        // In base ad iteration, modifica il poligono per cercare di recupeare
        // il lato del contour corretto
        let polygon = puzzle.get_polygon(&direction, iteration);
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
                        single_contours.push(Point::new(point.x, point.y));
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

    let up_left_point = match get_extreme(direction, &single_contours) {
        Ok((up_left_point, _)) => up_left_point,
        Err(err) => {
            println!(
                "Err split_single_contour -> get_extreme(up_left_point, _down_right_point): {:?}",
                err
            );
            return Err(anyhow!(err));
        }
    };

    let mut vector_after_first_traslate = Vector::new();
    for point in single_contours.iter() {
        vector_after_first_traslate.push(point - up_left_point);
    }

    let (up_left_point_translated, down_right_point_translated) = match get_extreme(
        direction,
        &vector_after_first_traslate,
    ) {
        Ok((val1, val2)) => (val1, val2),
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

    let mut countour_traslated = Vector::new();

    if y_max > 100 {
        for point in vector_rotated.iter() {
            countour_traslated.push(Point::new(point.x, -point.y));
        }
    } else {
        countour_traslated = vector_rotated.clone();
    }

    let (x_max, y_max, x_min, y_min) = find_limit(&countour_traslated);

    //let _ = cv::highgui::imshow("Phase", &phase);
    //draw_simple_contour(puzzle)?;
    //let key = cv::highgui::wait_key(500)?;
    //draw_contour(puzzle);

    Ok(SingleContourParams {
        single_contours,
        countour_traslated,
        x_max,
        x_min,
        y_min,
        y_max,
    })
}

fn draw_contour(puzzle: &PuzzlePiece) -> Result<(), anyhow::Error> {
    let mut phase = puzzle.original_image.clone();

    //draw_traslated_contour(puzzle, &mut phase)?;

    //draw_area_inside_polygon(puzzle, &mut phase)?;

    draw::draw_four_angles(&puzzle.corners, &mut phase)?;

    draw::draw_internal_contour(&puzzle.contours, &mut phase)?;

    println!(
        "Save image file {:?} in ./output/{}_contours.jpg",
        puzzle.get_file_name(),
        puzzle.get_output_file()
    );
    let _ = write_image(
        format!("./output/{}_contours.jpg", puzzle.get_output_file()).as_str(),
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
    let file_name_1 = puzzle1.get_file_name().to_string();
    let file_name_2 = puzzle2.get_file_name().to_string();
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
                        &file_name_1, &file_name_2, sequence1.dir, sequence2.dir,
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
                    sequence1.links.insert(file_name_2.clone(), sequence2.dir);
                    //sequence2.links.push(format!("{}-{:?}",puzzle1.file_name, sequence1.dir));
                    sequence2.links.insert(file_name_1.clone(), sequence1.dir);
                };
            }
        }
    }

    Ok((add, puzzle1, puzzle2))
}

fn main() {
    match my_contour() {
        Ok(_) => {
            //println!("End without errors");
        }
        Err(err) => println!("Error: {:?}", err),
    };
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
