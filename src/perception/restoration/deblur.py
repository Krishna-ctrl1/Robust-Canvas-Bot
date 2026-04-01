import cv2
import numpy as np

def get_motion_blur_kernel(size=15, angle=0):
    """
    Generate a simple motion blur kernel for simulating or deconvolving.
    """
    k = np.zeros((size, size))
    center = size // 2
    # Simple horizontal line; in reality, would rotate by angle
    k[center, :] = 1.0
    k = k / size
    return k

def wiener_deconvolution(image, kernel, noise_var=0.01):
    """
    Applies Wiener Deconvolution to a blurry image using a known PSF (kernel).
    image: 2D numpy array (grayscale)
    kernel: 2D numpy array (PSF)
    """
    # Pad kernel to match image
    dummy = np.zeros_like(image)
    kh, kw = kernel.shape
    dummy[:kh, :kw] = kernel
    
    # Shift kernel so center is at (0,0)
    dummy = np.roll(dummy, -kh//2, axis=0)
    dummy = np.roll(dummy, -kw//2, axis=1)
    
    # FFT
    IMG = np.fft.fft2(image)
    K = np.fft.fft2(dummy)
    
    # Wiener Filter
    K_conj = np.conj(K)
    W = K_conj / (np.abs(K)**2 + noise_var)
    
    # Apply filter
    RESTORED = IMG * W
    restored = np.fft.ifft2(RESTORED)
    restored = np.abs(restored)
    
    # Scale back to 0-255
    restored = np.clip(restored, 0, 255).astype(np.uint8)
    return restored

def enhance_blur(image):
    """
    Proxy function: De-blurs a BGR image by applying Wiener Deconvolution 
    to each channel assuming a generic horizontal motion blur.
    """
    if image is None:
        raise ValueError("Image is None")
        
    kernel = get_motion_blur_kernel(size=15) # Assume default size
    
    # Process per channel
    restored_channels = []
    for c in range(3):
        channel = image[:,:,c].astype(np.float64)
        restored_c = wiener_deconvolution(channel, kernel, noise_var=0.1)
        restored_channels.append(restored_c)
        
    restored_img = cv2.merge(restored_channels)
    
    # Apply a gentle sharpening unsharp mask as well for clearer edges
    gaussian_3 = cv2.GaussianBlur(restored_img, (0, 0), 2.0)
    unsharp_image = cv2.addWeighted(restored_img, 1.5, gaussian_3, -0.5, 0)
    
    return unsharp_image

if __name__ == "__main__":
    print("Deblurring module initialized.")
