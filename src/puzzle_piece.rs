use anyhow::anyhow;
use opencv::core::{Point, Vector};
use opencv::{self as cv, prelude::*};
use std::collections::HashMap;
use std::path::Path;

use crate::contour_with_dir::{ContourWithDir, Direction};
use crate::cv_utils::{get_first_contour, sub_process};

#[derive(Clone, Debug, Default)]
pub struct PuzzlePiece {
    file_name: String,
    output_file: String,
    pub contours: Vector<Vector<Point>>,
    pub contours_with_dir: Vec<ContourWithDir>,

    pub x_min: Point,
    pub y_min: Point,
    pub x_max: Point,
    pub y_max: Point,

    pub original_image: Mat,
    pub grey: Mat,
    pub original_contours: Vector<Vector<Point>>,
    pub corners: Vector<Point>,

    pub left_up_corner: Point,
    pub left_down_corner: Point,
    pub right_up_corner: Point,
    pub right_down_corner: Point,

    pub rect: cv::core::Rect,
    pub threshold: i32,
    pub center: Point,

    pub polygon: HashMap<Direction, Vector<Point>>,
    pub ok: bool,
    write_json: bool,
}

impl PuzzlePiece {
    pub fn new(file_name: &str) -> Self {
        Self {
            file_name: file_name.to_string(),
            output_file: Path::new(file_name)
                .file_stem()
                .unwrap()
                .to_str()
                .unwrap()
                .to_string(),
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

            left_up_corner: Point::new(i32::MAX, i32::MAX),
            left_down_corner: Point::new(i32::MAX, 0),
            right_up_corner: Point::new(0, i32::MAX),
            right_down_corner: Point::new(0, 0),

            rect: cv::core::Rect::default(),

            threshold: 0,
            center: Point::new(0, 0),
            polygon: HashMap::new(),
            ok: false,
            write_json: true,
        }
    }

    pub fn get_write_json(&self) -> bool {
        self.write_json
    }

    pub fn get_file_name(&self) -> &str {
        &self.file_name
    }

    pub fn get_output_file(&self) -> &str {
        &self.output_file
    }

    pub fn find_min_max(&mut self) {
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
    }

    pub fn set_corners(&mut self, corners: &Vector<Point>) {
        self.corners = corners.clone();

        for point in self.corners.iter() {
            if point.y < self.center.y {
                if point.x < self.center.x {
                    self.left_up_corner.x = point.x;
                    self.left_up_corner.y = point.y;
                } else {
                    self.right_up_corner.x = point.x;
                    self.right_up_corner.y = point.y;
                }
            } else if point.x < self.center.x {
                self.left_down_corner.x = point.x;
                self.left_down_corner.y = point.y;
            } else {
                self.right_down_corner.x = point.x;
                self.right_down_corner.y = point.y;
            }
        }
    }

    pub fn search_best_threshold(&mut self) -> Result<(), anyhow::Error> {
        let mut threshold = 0;
        let mut min_len = usize::MAX;
        for threshold_value in 160..230 {
            let contours = match sub_process(&self.grey, threshold_value) {
                Ok(contours) => contours,
                Err(err) => {
                    println!("search_best_threshold -> sub_process: {:?}", err);
                    return Err(anyhow!(err));
                }
            };

            let first = get_first_contour(&contours)?;

            if first.len() < min_len {
                min_len = first.len();
                threshold = threshold_value;
            }
        }

        self.threshold = threshold;
        Ok(())
    }

    pub fn read_image(&mut self) -> Result<(), anyhow::Error> {
        self.original_image = crate::cv_utils::read_image(&self.file_name)?;
        Ok(())
    }

    pub fn first_contour(&self) -> Result<Vector<Point>, anyhow::Error> {
        get_first_contour(&self.original_contours)
    }

    pub fn get_polygon(&self, direction: &Direction, iteration: i32) -> Vector<Point> {
        let mut polygon = Vector::new();
        let delta: i32 = 10;

        match direction {
            Direction::Down => {
                let delta1 = self.y_max.y + delta;
                polygon.push(self.left_down_corner);
                polygon.push(Point::new(self.left_down_corner.x, delta1));
                polygon.push(Point::new(self.right_down_corner.x, delta1));
                polygon.push(self.right_down_corner);
            }
            Direction::Up => {
                let delta1 = self.y_min.y - delta;
                polygon.push(self.left_up_corner);
                polygon.push(Point::new(self.left_up_corner.x, delta1));
                polygon.push(Point::new(self.right_up_corner.x, delta1));
                polygon.push(self.right_up_corner);
            }
            Direction::Right => {
                let delta1 = self.x_max.x + delta;
                polygon.push(self.right_up_corner);
                polygon.push(Point::new(delta1, self.right_up_corner.y));
                polygon.push(Point::new(delta1, self.right_down_corner.y));
                polygon.push(self.right_down_corner);
            }
            Direction::Left => {
                let delta1 = self.x_min.x - delta;
                polygon.push(self.left_up_corner);
                polygon.push(Point::new(delta1, self.left_up_corner.y));
                polygon.push(Point::new(delta1, self.left_down_corner.y));
                polygon.push(self.left_down_corner);
            }
        }

        let valore = delta * iteration * if iteration % 2 == 0 { 1 } else { -1 };
        let mut mid = self.center;
        match direction {
            Direction::Down | Direction::Up => {
                mid.y = self.center.y + valore;
            }
            Direction::Left | Direction::Right => {
                mid.x = self.center.x + valore;
            }
        }
        polygon.push(mid);

        polygon
    }
}
