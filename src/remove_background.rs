use std::path::PathBuf;

use frdm_tools::{conf::{Conf, DetectingContoursConf, FastScanConf, FineScanConf}, ContextRead, DetectingContoursCv, DetectingContoursCvCtx, Eval, Image, Initial, InitialCtx, Threshold};
use opencv::{
    core::{self, Vector}, highgui, imgcodecs, imgproc, prelude::*, Result
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
/// 
/// ### References:
/// [Adaptive background mixture models for real-time tracking](http://www.ai.mit.edu/projects/vsam/Publications/stauffer_cvpr98_track.pdf)
/// [OpenCV | Background Subtraction](https://docs.opencv.org/3.4/d8/d38/tutorial_bgsegm_bg_subtraction.html)
/// 
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
        highgui::named_window("BrightnessAndContrast", highgui::WINDOW_NORMAL)
            .map_err(|err| error.pass(err.to_string()))?;
        highgui::named_window("Result", highgui::WINDOW_NORMAL)
            .map_err(|err| error.pass(err.to_string()))?;
        
        let history = 100;
        let nmixtures = 5;
        let background_ratio = 0.01;
        let noise_sigma = 0.0;
        let learning_rate = 0.5;
        let mut fgbg = opencv::bgsegm::create_background_subtractor_mog(history, nmixtures, background_ratio, noise_sigma).unwrap();
        let conf = Conf {
            detecting_contours: DetectingContoursConf::default(),
            fast_scan: FastScanConf {
                geometry_defect_threshold: Threshold::min(),
            },
            fine_scan: FineScanConf {},
        };
        let fgmask = DetectingContoursCv::new(
            conf.detecting_contours.clone(),
            Initial::new(
                InitialCtx::new(),
            ),
        );

        let mut count = 0;
        loop {
            if count == 1 {
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
                            highgui::imshow("Frame", &img)
                                .map_err(|err| error.pass(err.to_string()))?;
                            let gamma = self.auto_gamma(&img);
                            let brc = self.auto_brightness_and_contrast(&gamma, Some(3.0))?;
                            let result = fgmask.eval(Image::new(brc.cols() as usize, brc.rows() as usize, brc.clone(), 0)).unwrap();
                            let result: &DetectingContoursCvCtx = result.read();
                            let result = &result.result.mat;
                            // let mut result = unsafe { Mat::new_rows_cols(gamma.rows(), gamma.cols(), gamma.typ()).unwrap() };
                            // fgbg.apply(&brc, &mut result, learning_rate)
                            //     .map_err(|err| error.pass(err.to_string()))?;
                            
                            // let mut result = unsafe { Mat::new_rows_cols(img.rows(), img.cols(), img.typ()).unwrap() };
                            // let images: opencv::core::Vector<Mat> = opencv::core::Vector::from_iter([
                            //      img,
                            //      fgmask,
                            // ]);
                            // opencv::core::hconcat(&images, &mut result)
                            //     .map_err(|err| error.pass(err.to_string()))?;
                            highgui::imshow("Gamma", &gamma)
                                .map_err(|err| error.pass(err.to_string()))?;
                            highgui::imshow("BrightnessAndContrast", &brc).unwrap();
                                // .map_err(|err| error.pass(err.to_string()))?;
                            highgui::imshow("Result", result)
                                .map_err(|err| error.pass(err.to_string()))?;
                            highgui::wait_key(100).unwrap();
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
    fn auto_brightness_and_contrast(&self, img: &Mat, clip_hist_percent: Option<f32>) -> Result<Mat, Error> {
        let mut clip_hist_percent = clip_hist_percent.unwrap_or(1.0);
        let dbg = "RemoveBackground";
        let error = Error::new(dbg, "auto_brightness_and_contrast");
        let mut gray = Mat::default();
        opencv::imgproc::cvt_color(img, &mut gray, imgproc::COLOR_BGR2GRAY, 0).unwrap();
        let gray_channels = gray.channels();
            // .map_err(|err| error.pass(err.to_string()))?;
        println!("{dbg}.evauto_brightness_and_contrastal | gray channels: {:?}", gray.channels());
        // highgui::imshow("BrightnessAndContrast", &gray).unwrap();
            // .map_err(|err| error.pass(err.to_string()))?;
        // highgui::wait_key(0).unwrap();

        // Grayscale histogram
        let mut hist = Mat::default();
        let hist_size = 256 as i32;
        let imgs: Vector<Mat> = Vector::from_iter([gray.clone()]);
        opencv::imgproc::calc_hist(
            &imgs,
            &Vector::from_slice(&[0]),
            &Mat::default(),
            &mut hist,
            &Vector::from_slice(&[hist_size]),
            &Vector::from_slice(&[0.0 ,256.0]),
            false,
        ).unwrap();
            // .map_err(|err| error.pass(err.to_string()))?;
        // let hist_size = hist.len();
        // println!("{dbg}.evauto_brightness_and_contrastal | hist: {:?}", hist);
        log::warn!("{dbg}.evauto_brightness_and_contrastal | hist_size: {:?}", hist_size);

        // highgui::imshow("BrightnessAndContrast", &hist).unwrap();
        // highgui::wait_key(0).unwrap();

        // Calculate cumulative distribution from the histogram
        let mut accumulator = vec![];
        accumulator.push(*hist.at::<f32>(0).unwrap());
        for index in 1..(hist_size as usize) {
            accumulator.push(accumulator[index -1] + *hist.at::<f32>(index as i32).unwrap())
        }
        // println!("{dbg}.evauto_brightness_and_contrastal | accumulator: {:?}", accumulator);

        // Locate points to clip
        let maximum = accumulator.last().unwrap();
        clip_hist_percent = clip_hist_percent * (maximum / 100.0);
        clip_hist_percent = clip_hist_percent / 2.0;
        println!("{dbg}.evauto_brightness_and_contrastal | maximum: {:?}", maximum);

        // Locate left cut
        let mut minimum_gray = 0;
        while accumulator[minimum_gray] < clip_hist_percent {
            minimum_gray += 1;
        }
        println!("{dbg}.evauto_brightness_and_contrastal | minimum_gray: {:?}", minimum_gray);

        // Locate right cut
        let mut maximum_gray = (hist_size - 1) as usize;
        while accumulator[maximum_gray] >= (maximum - clip_hist_percent) {
            maximum_gray -= 1;
        }
        println!("{dbg}.evauto_brightness_and_contrastal | maximum_gray: {:?}", maximum_gray);

        // Calculate alpha and beta values
        let alpha = 255.0 / ((maximum_gray - minimum_gray) as f64);
        let beta = - (minimum_gray as f64) * alpha;
        println!("{dbg}.evauto_brightness_and_contrastal | alpha: {},   beta: {}", alpha, beta);
        
        // Calculate new histogram with desired range and show histogram 
        // new_hist = cv2.calcHist([gray],[0],None,[256],[minimum_gray,maximum_gray])
        // plt.plot(hist)
        // plt.plot(new_hist)
        // plt.xlim([0,256])
        // plt.show()

        let mut dst = Mat::default();
        opencv::core::convert_scale_abs(img, &mut dst, alpha, beta).unwrap();
            // .map_err(|err| error.pass(err.to_string()))?;
        // return (auto_result, alpha, beta)
        Ok(dst)
    }
}