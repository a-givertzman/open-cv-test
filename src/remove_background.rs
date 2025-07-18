use std::path::PathBuf;

use opencv::{
    core,
    types,
    objdetect,
    highgui,
    imgcodecs,
    imgproc,
    videoio,
    Result,
    prelude::*,
    features2d,
};
use sal_core::error::Error;
///
/// This algorithm combines statistical background image estimation and per-pixel Bayesian segmentation.
/// 
/// It was introduced by Andrew B. Godbehere, Akihiro Matsukawa, and Ken Goldberg in their paper
/// "Visual Tracking of Human Visitors under Variable-Lighting Conditions for a Responsive Audio Art Installation" in 2012.
/// As per the paper, the system ran a successful interactive audio art installation called
/// “Are We There Yet?” from March 31 - July 31 2011 at the Contemporary Jewish Museum in San Francisco, California.
/// 
/// It uses first few (120 by default) frames for background modelling.
/// It employs probabilistic foreground segmentation algorithm that identifies possible foreground objects using Bayesian inference.
/// The estimates are adaptive; newer observations are more heavily weighted than old observations to accommodate variable illumination.
/// Several morphological filtering operations like closing and opening are done to remove unwanted noise.
/// You will get a black window during first few frames.
/// 
/// It would be better to apply morphological opening to the result to remove the noises. 
pub struct RemoveBackground {

}
//
//
impl RemoveBackground {
    pub fn new() -> Self {
        Self {  }
    }
    ///
    /// Performs algoritm
    pub fn eval(&self) -> Result<(), Error> {
        let dbg = "RemoveBackground";
        let error = Error::new(dbg, "eval");
        let dir = std::fs::read_dir("./assets/rope/").unwrap();
        let mut paths: Vec<PathBuf> = dir.map(|p| p.unwrap().path()).skip(5).collect();

        highgui::named_window("Frame", highgui::WINDOW_NORMAL)
            .map_err(|err| error.pass(err.to_string()))?;
        highgui::named_window("Gamma", highgui::WINDOW_NORMAL)
            .map_err(|err| error.pass(err.to_string()))?;
        highgui::named_window("Result", highgui::WINDOW_NORMAL)
            .map_err(|err| error.pass(err.to_string()))?;
        
        let history = 100;
        let nmixtures = 5;
        let background_ratio = 0.01;
        let noise_sigma = 0.0;
        let learning_rate = 0.5;
        let mut fgbg = opencv::bgsegm::create_background_subtractor_mog(history, nmixtures, background_ratio, noise_sigma).unwrap();
        let mut count = 0;
        loop {
            if count == 100 {
                let dir = std::fs::read_dir("./assets/rope/").unwrap();
                paths = dir.map(|p| p.unwrap().path()).skip(0).collect()
            };
            count += 1;
            for path in paths.iter().cycle() {
                if path.is_file() {
                    log::debug!("{dbg}.eval | path: {}", path.display());
                    match imgcodecs::imread(&path.to_str().unwrap(), imgcodecs::IMREAD_COLOR) {
                        Ok(img) => {
                            log::debug!("{dbg}.eval | file read successfully: {:?}", img.size());
                            let gamma = self.auto_gamma(&img);
                            let mut fgmask = unsafe { Mat::new_rows_cols(gamma.rows(), gamma.cols(), gamma.typ()).unwrap() };
                            fgbg.apply(&gamma, &mut fgmask, learning_rate)
                                .map_err(|err| error.pass(err.to_string()))?;
                            
                            // let mut result = unsafe { Mat::new_rows_cols(img.rows(), img.cols(), img.typ()).unwrap() };
                            // let images: opencv::core::Vector<Mat> = opencv::core::Vector::from_iter([
                            //      img,
                            //      fgmask,
                            // ]);
                            // opencv::core::hconcat(&images, &mut result)
                            //     .map_err(|err| error.pass(err.to_string()))?;
                            highgui::imshow("Frame", &img)
                                .map_err(|err| error.pass(err.to_string()))?;
                            highgui::imshow("Gamma", &gamma)
                                .map_err(|err| error.pass(err.to_string()))?;
                            highgui::imshow("Result", &fgmask)
                                .map_err(|err| error.pass(err.to_string()))?;
                            highgui::wait_key(1).unwrap();
                        },
                        Err(err) => log::warn!("{dbg}.eval | Read file '{}' error: {:?}", path.display(), err),
                    };
                }
            }        
        }
        Ok(())
    }
    ///
    /// Step 1: Gamma correction
    ///
    /// The reasoning of this step is to balance out the contrast of the whole image
    /// (since your image can be slightly overexposed/underexposed depending to the lighting condition).
    ///
    /// This may seem at first as an unnecessary step, but the importance of it cannot be underestimated:
    /// in a sense, it normalizes the images to the similar distributions of exposures,
    /// so that you can choose meaningful hyper-parameters later (e.g. the DELTA parameter in next section,
    /// the noise filtering parameters, parameters for morphological stuffs, etc.)
    fn auto_gamma(&self, img: &Mat) -> Mat {
        // build a lookup table mapping the pixel values [0, 255] to
        // their adjusted gamma values
        let mid = 0.5f64;
        let mean = opencv::core::mean(&img, &Mat::default()).unwrap().into_iter().take(3).map(|v| v as f64).sum::<f64>() / 3.0;
        println!("mean: {:?}", mean);
        let gamma: f64 = (mid * 255.0).ln()/mean.ln();
        println!("gamma: {:?}", gamma);
        let inv_gamma = 1.0 / gamma;
        let table: Vec<_> = (0..256).map(|i| (255.0 * ((i as f64 / 255.0).powf(inv_gamma))) as u8 ).collect();
        // println!("table: {:?}", table);
        let mut dst = Mat::default();
        opencv::core::lut(&img, &Mat::from_slice(&table).unwrap(), &mut dst).unwrap();
        dst
    }
}