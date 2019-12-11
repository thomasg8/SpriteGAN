import os, cv2, imutils, json, numpy as np, networkx as nx, skimage.io
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
from skimage.measure import compare_ssim
from datetime import datetime
from statistics import mean
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, LeakyReLU, Reshape, Conv2DTranspose



def auto_canny(img):
    """Calculates and executes proper threshold for edge detection (Canny)
    Parameters:
        img: image object
        sigman: thresold caluclator constant
    Returns:
        image post Canny
    """
    v = np.median(img)
    lower = int(max(0, (1.0 - 0.33) * v))
    upper = int(min(255, (1.0 + 0.33) * v))
    cnt = cv2.Canny(img, lower, upper)
    return cnt

def crop_canny(img):
    """Crops base image based on edge Canny values
    Parameters:
        img: image object
    Returns:
        cropped img
    """
    cnt = auto_canny(img)
    pts = np.argwhere(cnt>0)
    y1,x1 = pts.min(axis=0)
    y2,x2 = pts.max(axis=0)
    return img[y1:y2, x1:x2]

def process_image(src, grayscale=False, resize_val=(32,32)):
    """
    Reads image, Crops image based on optimized Canny, Resizes image
    Parameters:
        src: origin filepath
        grayscale: determines color channels
    Returns:
        img
    """
    if not grayscale:
        img = cv2.imread(src)
    else:
        img = cv2.imread(src, cv2.IMREAD_GRAYSCALE)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.shape[:2]!=resize_val:
        cnt = crop_canny(img)
        img32 = cv2.resize(cnt, resize_val, interpolation = cv2.INTER_AREA)
        return img, img32
    else:
        return img, img
def get_sorted_collections(mappings):
    """For the collections I have manually sorted, organizes them by origin
        Parameters:
            None
        Returns:
            Sorted dictionary of collections and file ids
    """
    sorted_collections = list(set(os.listdir('SpriteFiles2DBase')))
    sortedidd = {}
    for k, v in mappings['filepath_id'].items():
        kc = k.split('\\')[1]
        if kc in sorted_collections:
            if kc not in sortedidd:
                sortedidd[kc] = [v]
            else:
                sortedidd[kc].append(v)
    return sortedidd

def gen_similarity_d(sortedidd, imgs32, srcs, classif): #include class
    """Goes through each class/collection pair and calculates the similarity to 200 neighboring images
        Parameters:
            sortedidd: sorted collection files
            imgs32: list of imagefiles
            srcs: list of filepaths
            classif: Classification

        Returns:
            dictionary of collcetion and similarity pairs
    """
    print("Creating Similarity Dict for "+ classif)
    similarity_d = {}
    for collection, files in sortedidd.items():
        similarity_d[collection] = []; pairs = []
        for i1 in range(len(files)):

            if files[i1]+'.png' in srcs and files[i1]+'.png' in os.listdir('SpriteFiles2DSorted//'+classif):
                i_a = srcs.index(files[i1]+'.png'); img1 = imgs32[i_a]
                if i1+200>len(files):
                    end = len(files)
                else:
                    end = i1+200
                for i2 in range(i1+1, end):
                    if files[i2]+'.png' in srcs and files[i1]+'.png' in os.listdir('SpriteFiles2DSorted//'+classif):
                        i_b = srcs.index(files[i2]+'.png'); img2 = imgs32[i_b]
                        score = compare_ssim(img1, img2, multichannel=True) # multichannel for RGB images
                        if score>.90:
                            pairs.append((i_a,i_b))
        similarity_d[collection] = pairs

    return similarity_d

def get_removal_list(similarity_d, collection):
    """ Removes nodes with too many similar images from most connections to least, leaving no connections
        Parameters:
            similarity_d: dictionary of similarity scores
            collection: image collection
        Returns:
            list of too similar imgs
    """
    G=nx.Graph(); originalG = nx.Graph()

    G.add_edges_from(similarity_d[collection])
    originalG.add_edges_from(similarity_d[collection])

    connections = [len(G[node]) for node in G.nodes] #initializing
    while max(connections)!=0 and len(G.nodes)!=1:
        connections = [len(G[node]) for node in G.nodes]
        biggest_offender = list(G.nodes)[connections.index(max(connections))]
        G.remove_node(biggest_offender) #removes node with most similarity to others
    #print(len(originalG.nodes), 'to', len(G.nodes))
    too_similar = list(set(originalG.nodes) - set(G.nodes))
    return too_similar

def get_too_similar_list(active, imgs32, srcs, mappings):
    """Generates list of files to not include in the cnn due to similarity
    Parameters:
        active: classification active
        imgs32: image list
        srcs: filepath list
    Returns:
        list of indices to remove
    """
    too_similar = []
    sorted_collections = get_sorted_collections(mappings)
    for classif in active:
        similarity_d = gen_similarity_d(sorted_collections, imgs32, srcs, classif)
        for collection, v in similarity_d.items():
            if len(v)>1:
                ts = get_removal_list(similarity_d, collection)
                too_similar+=ts

    return too_similar
# Define GAN
def gen_discriminator(input_shape = (32,32,3)):
    model = Sequential()
    model.add(Conv2D(64, 5, strides=2, padding='same', input_shape=input_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Conv2D(128, 5, strides=2, padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Conv2D(256, 5, strides=2, padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5), metrics=['accuracy'])
    return model

def define_generator(ld=100):
    """ld: latentent space input dimension"""
    model = Sequential()
    n_nodes = 256 * 4 * 4
    model.add(Dense(n_nodes, input_dim=ld))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((4, 4, 256))) #4x4

    model.add(Conv2DTranspose(128, 4, strides=2, padding='same')) #8x8
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2DTranspose(128, 4, strides=2, padding='same')) #16x16
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2DTranspose(128, 4, strides=2, padding='same')) #32x32
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(3, 3, activation='tanh', padding='same'))
    return model

def gen_gan(generator, discriminator):
    discriminator.trainable = False #makes weights constant
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model
def gen_real_fake_imgs(generator, imgs, n):
    """
    Generates arrays of real and fake imgs.
    Parameters:
        imgs: array of normalized images
        n: number of samples
        generator: generator model
    Returns:
        real and fake img arrays and classification arrays
    """
    # Randomly selects real samples
    real_X = imgs[np.random.randint(0, imgs.shape[0], n)]
    real_y = np.ones((n, 1))

    x_input = np.random.randn(100, n).reshape(n, 100)
    fake_X = generator.predict(x_input)
    fake_y = np.zeros((n, 1))

    return real_X, real_y, fake_X, fake_y
def save_imgs(imgs, e):
    """ Saves created images for manual identification
        Parameters:
            e: epoch
            imgs: list of created images
        Returns:
            None"""
    imgs = (imgs + 1) / 2.0 # rescale
    for i in range(10):
 	pyplot.imshow(imgs[i])
	pyplot.axis('off')
        pyplot.savefig('GeneratedImages32/{}/{}.png'.format(e, i),  bbox_inches='tight')

def performance(e, generator, discriminator, imgs, ld=100, n=256):
    real_X, real_y, fake_X, fake_y = gen_real_fake_imgs(generator, imgs, int(n/2))
    _, fake_acc = discriminator.evaluate(real_X, real_y, verbose=0)
    _, real_acc = discriminator.evaluate(fake_X, fake_y, verbose=0)

    print("On {}, Real Acc: {}. Fake Acc: {}.".format(e, int(real_acc*100), int(fake_acc*100)))
    os.makedirs('GeneratedImages32/{}'.format(e))
    save_imgs(fake_X, e)
    generator.save('GeneratedImages32/{}/generator.h5'.format(e))
    discriminator.save('GeneratedImages32/{}/discriminator.h5'.format(e))



def train(generator, discriminator, gan, imgs, ld=100, epochs=100, n=256):
    data = []
    per_e = int(len(imgs)/n)
    if per_e<256:
        per_e=256
    for i in range(epochs):
        for i2 in range(per_e):
            real_X, real_y, fake_X, fake_y = gen_real_fake_imgs(generator, imgs, int(n/2))
            discrim_loss_A, _ = discriminator.train_on_batch(real_X, real_y)
            discrim_loss_B, _ = discriminator.train_on_batch(fake_X, fake_y)

            X_gan = np.random.randn(ld, n).reshape(n, ld)
            y_gan = np.ones((n, 1))
            gener_loss_A = gan.train_on_batch(X_gan, y_gan)
            data.append([i, i2+1, str(discrim_loss_A), str(discrim_loss_B), str(gener_loss_A), str(datetime.now())])
            print("{}/{}, {}/{}, {}, {}, {}".format(i, epochs, i2+1, per_e, round(discrim_loss_A, 3), round(discrim_loss_B, 3), round(gener_loss_A,3)))
        if i%10 == 0:
            performance(i, generator, discriminator, imgs)
    return data
def main():
    paths = {}
    with open('mappings.json') as f:
        mappings = json.load(f)
    # active = ['Animated', 'Character', 'Item', 'SpriteSheets', 'TokensIcon', 'Vehicles', 'BuildingEnv']
    active = ['Animated'] #Character
    grayscale_boolean = False
    imgs = []; imgs32 = []; fails = []; cats = []; srcs=[]
    for folder in active:
        for filepath in os.listdir('SpriteFiles2DSorted/'+folder):
            try:
                img, img32 = process_image('SpriteFiles2DSorted/'+folder+'/'+filepath, grayscale=grayscale_boolean)
                imgs.append(img); imgs32.append(img32)
                cats.append(folder); srcs.append(filepath)
            except:
                fails.append('SpriteFiles2DSorted/'+folder+'/'+filepath)

    try:
    	with open('too_similar_{}.json'.format(active[0].lower())) as file:
      	    too_similar = json.load(file)
    except:
	too_similar = get_too_similar_list(active, imgs32, srcs, mappings)
    # removes indices from lists
    imgs32_div = imgs32
    cats_div = cats
    for index in sorted(too_similar, reverse=True):
        del imgs32_div[index]
        del cats_div[index]
    imgs32_div_norm = [img.astype('float32')/255 for img in imgs32_div]
    imgs32_div_norm = np.asarray(imgs32_div_norm)

    discriminator = gen_discriminator()
    discriminator.summary(); print('\n')
    generator = define_generator()
    generator.summary(); print('\n')
    gan = gen_gan(generator, discriminator)
    gan.summary()



    training_data = train(generator, discriminator, gan, imgs32_div_norm, epochs=500)
    with open('GeneratedImages32/training_data.txt', 'w') as outfile:
        json.dump(training_data, outfile)

main()
