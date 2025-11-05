import os
import glob
from tqdm import tqdm
import tensorflow.compat.v1 as tf
import numpy as np
import cv2
import logging
from datetime import datetime
import sys

import unprocess
from process_new import process_to_linear_rgb, apply_gamma_compression


# --- TUNABLE PARAMETERS ---
ILLUMINATION_DARKEN_FACTOR = 0.01
NOISE_LEVEL = 0.0001
CONTRAST_STRENGTH = 0.8
SATURATION_BOOST = 1.2
# --- End Configuration ---

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = 'generation.log'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, mode='w')
    ]
)

logger = logging.getLogger(__name__)
logger.info(f"ILLUMINATION_DARKEN_FACTOR: {ILLUMINATION_DARKEN_FACTOR}")
logger.info(f"NOISE_LEVEL: {NOISE_LEVEL}")
logger.info(f"CONTRAST_STRENGTH: {CONTRAST_STRENGTH}")
logger.info(f"SATURATION_BOOST: {SATURATION_BOOST}")

def adjust_linear_saturation(linear_image, saturation_factor):
    """Adjusts saturation on a linear RGB image."""
    luminance = (linear_image[..., 0] * 0.2126 +
                 linear_image[..., 1] * 0.7152 +
                 linear_image[..., 2] * 0.0722)
    luminance = tf.expand_dims(luminance, axis=-1)
    
    saturated_image = luminance + (linear_image - luminance) * saturation_factor
    return tf.clip_by_value(saturated_image, 0.0, 1.0)

def apply_s_curve_contrast(linear_image, strength=1.0):
    """
    Applies a contrast-enhancing S-curve.
    A strength of 0.0 is no change, 1.0 is a strong curve.
    """
    s_curve_image = 3.0 * linear_image**2 - 2.0 * linear_image**3
    
    blended_image = linear_image * (1.0 - strength) + s_curve_image * strength
    return tf.clip_by_value(blended_image, 0.0, 1.0)

def build_lowlight_graph(source_path_tensor, target_path_tensor):
    image_gt_bytes = tf.io.read_file(source_path_tensor)
    image_gt_tensor = tf.image.decode_image(image_gt_bytes, channels=3)
    image_gt_tensor = tf.cast(image_gt_tensor, tf.float32) / 255.0
    
    raw_image, metadata = unprocess.unprocess(image_gt_tensor)
    
    shot_noise, read_noise = unprocess.get_noise_levels(NOISE_LEVEL)
    noisy_raw_image = unprocess.add_noise(raw_image, shot_noise, read_noise)
    
    noisy_raw_image_batched = tf.expand_dims(noisy_raw_image, axis=0)
    red_gain_batched = tf.expand_dims(metadata['red_gain'], axis=0)
    blue_gain_batched = tf.expand_dims(metadata['blue_gain'], axis=0)
    cam2rgb_batched = tf.expand_dims(metadata['cam2rgb'], axis=0)

    linear_rgb_batched = process_to_linear_rgb(noisy_raw_image_batched,
                                               red_gain_batched,
                                               blue_gain_batched,
                                               cam2rgb_batched)
    linear_rgb = tf.squeeze(linear_rgb_batched, axis=0)

    def py_bilateral_filter(image_np):
        return cv2.bilateralFilter(image_np.numpy().astype(np.float32), d=31, sigmaColor=0.1, sigmaSpace=8.0)

    base_illumination = tf.py_function(
        py_bilateral_filter,
        inp=[linear_rgb],
        Tout=tf.float32
    )
    base_illumination.set_shape(linear_rgb.shape)

    detail_layer = linear_rgb / (base_illumination + 1e-8)

    dark_illumination = base_illumination * ILLUMINATION_DARKEN_FACTOR

    low_light_linear = dark_illumination * detail_layer

    image_with_contrast = apply_s_curve_contrast(low_light_linear, strength=CONTRAST_STRENGTH)

    final_linear_image = adjust_linear_saturation(image_with_contrast, SATURATION_BOOST)

    final_srgb_image = apply_gamma_compression(final_linear_image)
    
    final_image_display = tf.cast(tf.clip_by_value(final_srgb_image, 0.0, 1.0) * 255.0, tf.uint8)
    encoded_image = tf.image.encode_jpeg(final_image_display)
    
    return tf.io.write_file(target_path_tensor, encoded_image)


if __name__ == '__main__':

    SOURCE_DIR = "../Lowlightdataset/coco_original"
    TARGET_DIR = "../Lowlightdataset/coco_dark_raw"

    os.makedirs(TARGET_DIR, exist_ok=True)
    logger.info(f"SOURCE_DIR: {SOURCE_DIR}")
    
    source_images = glob.glob(os.path.join(SOURCE_DIR, '*.jpg')) + \
                    glob.glob(os.path.join(SOURCE_DIR, '*.png'))
    
    logger.info(f"Found {len(source_images)} images to process in {SOURCE_DIR}.")

    for source_path in tqdm(source_images, desc=f"Generating low-light images"):
        filename = os.path.basename(source_path)
        target_path = os.path.join(TARGET_DIR, filename)

        if os.path.exists(target_path):
            continue
        
        build_lowlight_graph(tf.constant(source_path), tf.constant(target_path))

    print(f"Data generation complete!")