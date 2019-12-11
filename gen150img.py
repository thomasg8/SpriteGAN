from tensorflow.keras.models import load_model
import cv2, os, numpy as np

def main():
    """Generates 150 images from the GAN generator"""
    try:
        os.makedirs('GeneratedImages') #location of generated images
        start = 0 # for file naming
    except:
        start = len(os.listdir('GeneratedImages'))

    generator = load_model('generator.h5')
    x_input = np.random.randn(100, 150).reshape(150, 100) #change 150 if you want more
    imgs = generator.predict(x_input) # actual image generation
    for i in range(100):
        img = cv2.cvtColor(imgs[i], cv2.COLOR_BGR2RGB) #images generated as BGR
        cv2.imwrite('GeneratedImages/{}.png'.format(start + i), img*256)

main()
