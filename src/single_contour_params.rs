use opencv::core::Point;
use opencv::core::Vector;

pub struct SingleContourParams {
    pub single_contours: Vector<Point>,
    pub countour_traslated: Vector<Point>,
    pub x_max: i32,
    pub y_min: i32,
    pub y_max: i32,
}
