use opencv::core::{Point, Vector};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use strum_macros::EnumIter;

#[derive(EnumIter, Debug, Hash, Eq, PartialEq, Clone, Copy, Serialize, Deserialize)]
pub enum Direction {
    Up,
    Down,
    Right,
    Left,
}

#[derive(EnumIter, Debug, Hash, Eq, PartialEq, Clone, Copy, Serialize, Deserialize)]
pub enum Genders {
    Unknown,
    Female,
    Male,
    Line,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ContourWithDir {
    #[serde(skip)]
    pub countour: Vector<Point>,
    #[serde(skip)]
    pub countour_traslated: Vector<Point>,
    pub dir: Direction,
    pub gender: Genders,
    pub x_max: i32,
    pub y_min: i32,
    pub y_max: i32,
    pub d1: i32,
    pub d2: i32,
    pub d3: i32,
    pub d4: i32,
    pub d5: i32,
    #[serde(default)]
    pub links: HashMap<String, Direction>,
}

impl ContourWithDir {
    pub fn new(
        countour: Vector<Point>,
        dir: Direction,
        gender: Genders,
        countour_traslated: Vector<Point>,
        x_max: i32,
        y_min: i32,
        y_max: i32,
    ) -> Self {
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
            links: HashMap::new(),
        }
    }

    pub fn dx(&mut self) {
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

    pub fn clear_links(&mut self) {
        self.links.clear();
    }
}

pub fn get_extreme(
    direction: Direction,
    vector: &Vector<Point>,
) -> Result<(Point, Point), anyhow::Error> {
    let mut up_left_point = Point::new(0, 0);
    let mut down_right = Point::new(0, 0);
    match direction {
        Direction::Down | Direction::Up => {
            let mut x_min = i32::MAX;
            let mut x_max = 0;

            for index in 0..vector.len() {
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
        }
        Direction::Left | Direction::Right => {
            let mut y_min = i32::MAX;
            let mut y_max = 0;

            for index in 0..vector.len() {
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
        }
    }
    Ok((up_left_point, down_right))
}
