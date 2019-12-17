# Using GANs for Sprite Generation
A Generative Adversarial Network for creating character sprites. Please read SpriteGAN.pdf if you are interested in finding out more.

## Overview
This paper proposes the use of a Generative Adversarial Network (GAN) to generate novel images of video game sprite characters. The generation of unique characters in video games could make the gaming experience unique to each individual playing the game which could decrease potential boredom due to repetition. GANs are composed of two adversarial models, the generator and the discriminator, which compete against each other in a zero sum game until Nash's equilibrium is reached or training is terminated. The GAN in this paper has a standard architecture. In this paper, the discriminator is a convolutional neural network and the generator is a transposed convolutional network. This model was trained on 4,272 character images collected from Open Game Art using web scraping. In total, this dataset contains 16,627 CC0 licensed images of sprites from 284 collections. These images were manually sorted into 5 categories: Characters, Items, Animated, Vehicles, and Tokens/Icons. 

<a href="url"><img src="https://github.com/thomasg8/SpriteGAN/blob/master/Code/Figs/Cleaning.png" align="right" height="256" width="256" ></a>

Because the data was scraped from 284 collections, there were two major data impurities: similar images and different image sizes. As the sprites are meant for video games, many artists provided their character sprites doing various actions, e.g. swinging a sword. To not overfit to these character sprites, similar images were removed from the training set using a structural similarity index calculation. Images in each collection were converted to a network, with connections being the similarity index between two images. Then images were iteratively removed with similarity connections greater than 0.9 out of 1.0. To fit to the 32x32x3 size requirement for the GAN, many images required major resizing. To avoid loss of detail due to downsizing, the images were first cropped using the Canny edge detection algorithm. Then the images were downsized using the inter area interpolation method. This is shown in the figure to the right. 

These images were passed through a conventional generator and discriminator architecture. The objective creativity of the generated sprites was quantitatively analyzed with a IRB certified survey of 40 participants. The results showed that most participants were not able to accurately distinguish between human and computer generated sprites. In addition, it shows that the participants were inherently distrustful of all images, classifying significantly more images as computer generated, as opposed to human generated. The images (32x32x3) used in the survery are displayed below.

<a href="url"><img src="https://github.com/thomasg8/SpriteGAN/blob/master/Code/Figs/Survey.PNG" align="center"></a>

## Generating Images
To generate character sprites for yourself, run gen150img.py. It will create a folder and generate 150 sprites. 
