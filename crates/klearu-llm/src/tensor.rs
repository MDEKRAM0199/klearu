/// Zero-copy 2D view over a flat `&[f32]` slice with shape `[rows, cols]`.
#[derive(Debug, Clone, Copy)]
pub struct Tensor2<'a> {
    data: &'a [f32],
    rows: usize,
    cols: usize,
}

impl<'a> Tensor2<'a> {
    pub fn new(data: &'a [f32], rows: usize, cols: usize) -> Self {
        assert_eq!(
            data.len(),
            rows * cols,
            "Tensor2: data len {} != rows {} * cols {}",
            data.len(),
            rows,
            cols
        );
        Self { data, rows, cols }
    }

    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn cols(&self) -> usize {
        self.cols
    }
    pub fn data(&self) -> &'a [f32] {
        self.data
    }

    pub fn row(&self, i: usize) -> &'a [f32] {
        let start = i * self.cols;
        &self.data[start..start + self.cols]
    }

    pub fn get(&self, row: usize, col: usize) -> f32 {
        self.data[row * self.cols + col]
    }
}

/// Zero-copy mutable 2D view over a flat `&mut [f32]` slice.
#[derive(Debug)]
pub struct Tensor2Mut<'a> {
    data: &'a mut [f32],
    rows: usize,
    cols: usize,
}

impl<'a> Tensor2Mut<'a> {
    pub fn new(data: &'a mut [f32], rows: usize, cols: usize) -> Self {
        assert_eq!(
            data.len(),
            rows * cols,
            "Tensor2Mut: data len {} != rows {} * cols {}",
            data.len(),
            rows,
            cols
        );
        Self { data, rows, cols }
    }

    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn row(&self, i: usize) -> &[f32] {
        let start = i * self.cols;
        &self.data[start..start + self.cols]
    }

    pub fn row_mut(&mut self, i: usize) -> &mut [f32] {
        let start = i * self.cols;
        &mut self.data[start..start + self.cols]
    }

    pub fn get(&self, row: usize, col: usize) -> f32 {
        self.data[row * self.cols + col]
    }

    pub fn set(&mut self, row: usize, col: usize, val: f32) {
        self.data[row * self.cols + col] = val;
    }

    pub fn as_ref(&self) -> Tensor2<'_> {
        Tensor2 {
            data: self.data,
            rows: self.rows,
            cols: self.cols,
        }
    }

    pub fn data_mut(&mut self) -> &mut [f32] {
        self.data
    }
}

/// Zero-copy 3D view over a flat `&[f32]` slice with shape `[d0, d1, d2]`.
#[derive(Debug, Clone, Copy)]
pub struct Tensor3<'a> {
    data: &'a [f32],
    d0: usize,
    d1: usize,
    d2: usize,
}

impl<'a> Tensor3<'a> {
    pub fn new(data: &'a [f32], d0: usize, d1: usize, d2: usize) -> Self {
        assert_eq!(
            data.len(),
            d0 * d1 * d2,
            "Tensor3: data len {} != {} * {} * {}",
            data.len(),
            d0,
            d1,
            d2
        );
        Self { data, d0, d1, d2 }
    }

    pub fn d0(&self) -> usize {
        self.d0
    }
    pub fn d1(&self) -> usize {
        self.d1
    }
    pub fn d2(&self) -> usize {
        self.d2
    }

    /// Get a 2D slice along the first axis: `self[i, :, :]`.
    pub fn slice_d0(&self, i: usize) -> Tensor2<'a> {
        let start = i * self.d1 * self.d2;
        let end = start + self.d1 * self.d2;
        Tensor2::new(&self.data[start..end], self.d1, self.d2)
    }

    pub fn get(&self, i: usize, j: usize, k: usize) -> f32 {
        self.data[i * self.d1 * self.d2 + j * self.d2 + k]
    }
}

/// Zero-copy mutable 3D view.
#[derive(Debug)]
pub struct Tensor3Mut<'a> {
    data: &'a mut [f32],
    d0: usize,
    d1: usize,
    d2: usize,
}

impl<'a> Tensor3Mut<'a> {
    pub fn new(data: &'a mut [f32], d0: usize, d1: usize, d2: usize) -> Self {
        assert_eq!(
            data.len(),
            d0 * d1 * d2,
            "Tensor3Mut: data len {} != {} * {} * {}",
            data.len(),
            d0,
            d1,
            d2
        );
        Self { data, d0, d1, d2 }
    }

    pub fn d0(&self) -> usize {
        self.d0
    }
    pub fn d1(&self) -> usize {
        self.d1
    }
    pub fn d2(&self) -> usize {
        self.d2
    }

    /// Get a mutable 2D slice along the first axis: `self[i, :, :]`.
    pub fn slice_d0_mut(&mut self, i: usize) -> Tensor2Mut<'_> {
        let start = i * self.d1 * self.d2;
        let end = start + self.d1 * self.d2;
        Tensor2Mut::new(&mut self.data[start..end], self.d1, self.d2)
    }

    pub fn get(&self, i: usize, j: usize, k: usize) -> f32 {
        self.data[i * self.d1 * self.d2 + j * self.d2 + k]
    }

    pub fn set(&mut self, i: usize, j: usize, k: usize, val: f32) {
        self.data[i * self.d1 * self.d2 + j * self.d2 + k] = val;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor2_indexing() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = Tensor2::new(&data, 2, 3);
        assert_eq!(t.rows(), 2);
        assert_eq!(t.cols(), 3);
        assert_eq!(t.row(0), &[1.0, 2.0, 3.0]);
        assert_eq!(t.row(1), &[4.0, 5.0, 6.0]);
        assert_eq!(t.get(1, 2), 6.0);
    }

    #[test]
    fn test_tensor2_mut() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0];
        let mut t = Tensor2Mut::new(&mut data, 2, 2);
        t.set(0, 1, 99.0);
        assert_eq!(t.get(0, 1), 99.0);
        t.row_mut(1).copy_from_slice(&[10.0, 20.0]);
        assert_eq!(t.row(1), &[10.0, 20.0]);
    }

    #[test]
    fn test_tensor3_indexing() {
        let data: Vec<f32> = (0..24).map(|x| x as f32).collect();
        let t = Tensor3::new(&data, 2, 3, 4);
        assert_eq!(t.d0(), 2);
        assert_eq!(t.d1(), 3);
        assert_eq!(t.d2(), 4);
        assert_eq!(t.get(0, 0, 0), 0.0);
        assert_eq!(t.get(1, 2, 3), 23.0);
        let slice = t.slice_d0(1);
        assert_eq!(slice.rows(), 3);
        assert_eq!(slice.cols(), 4);
        assert_eq!(slice.get(0, 0), 12.0);
    }

    #[test]
    fn test_tensor3_mut() {
        let mut data = vec![0.0; 12];
        let mut t = Tensor3Mut::new(&mut data, 2, 3, 2);
        t.set(1, 2, 1, 42.0);
        assert_eq!(t.get(1, 2, 1), 42.0);
    }

    #[test]
    #[should_panic(expected = "Tensor2: data len")]
    fn test_tensor2_bad_shape() {
        let data = vec![1.0, 2.0, 3.0];
        Tensor2::new(&data, 2, 2);
    }

    #[test]
    #[should_panic(expected = "Tensor3: data len")]
    fn test_tensor3_bad_shape() {
        let data = vec![1.0; 10];
        Tensor3::new(&data, 2, 3, 2);
    }
}
