use std::fs;

pub fn find_files(path: &str) -> Vec<String> {
    fs::read_dir(path)
        .map(|entries| {
            entries
                .filter_map(Result::ok)
                .filter(|e| e.path().extension().is_some_and(|ext| ext == "jpg"))
                .filter_map(|e| e.path().to_str().map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_else(|_| {
            fs::read_dir(path)
                .into_iter()
                .for_each(|e| println!("read_dir finds {:?}", e));
            println!("find_files return empty list inside \"{}\" folder", path);
            Vec::new()
        })
}
