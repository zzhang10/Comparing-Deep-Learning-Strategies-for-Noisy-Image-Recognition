
# For building neural networks
import tensorflow as tf 
from tensorflow.keras import optimizers, regularizers
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, LeakyReLU, Conv2D, MaxPooling2D, Flatten
from keras.layers import Input, UpSampling2D, BatchNormalization, Reshape

# For data manipulation
import numpy as np 
import pandas as pd

# For Generating figures
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

# For adding noise
import cv2
from skimage.util import random_noise

# For reducing RAM usage
import gc

# For dataset analysis
from sklearn.manifold import TSNE



class ImageClassifier:
    ''' 
    The ImageClassifier class builds, trains and tests deep learning models including the
    Multilayered Perceptron (MLP), Convolutional Neural Network(CNN), Stacked Denoising
    Autoencoder (SDAE) and Convolutional Denoising Autoencoder (CDAE) to compare accuracy 
    on MNIST, Fashion MNIST (FMNIST), and CIFAR-10 datasets under noiseless or noisy conditions.
    It also preprocesses, reshapes and analyzes data, and produces graph for hyperparameter
    tuning and final result comparisons.
    '''


    def __init__(self):
        '''
        Well, here is a constructor. 
        '''
        # Raw, noiseless training and testing sets downloaded
        self.train_sample = None
        self.test_sample = None

        # Training and testing sets after adding noise
        self.noisy_train_sample = None
        self.noisy_test_sample = None

        # NN for denoising and classifying
        self.denoiser = None
        self.classifier = None

        # Indicates type of dataset 
        self.dataset = None

        # Indicates dimensions for MLP and CNN
        self.MLP_dim = None
        self.CNN_dim = None

        # Post-processed datasets, specialized for training and testing on a model
        self.x_train_MLP = None
        self.x_train_CNN = None
        self.x_train_noisy_MLP = None
        self.x_train_noisy_CNN = None
        self.x_test_MLP = None
        self.x_test_CNN = None
        self.x_test_noisy_MLP = None
        self.x_test_noisy_CNN = None
        self.y_train_onehot = None
        self.y_test_onehot = None

        # Constant table for label names
        self.label_names = {
            "MNIST" : {0:"0", 1:"1", 2:"2", 3:"3", 4:"4", 5:"5", 6:"6", 7:"7", 8:"8",9:"9"},
            "FMNIST" : {0:"T-shirt", 1:"Trouser", 2:"Pullover", 3:"Dress", 4:"Coat", 5:"Sandal", 
                        6:"Shirt", 7:"Sneaker", 8:"Bag", 9:"Ankle boot"},
            "CIFAR10" : {0:"Airplane", 1:"Automobile", 2:"Bird", 3:"Cat", 4:"Deer", 5:"Dog", 
                        6:"Frog", 7:"Horse", 8:"Ship", 9:"Truck"}
        }


    def data_summary(self, added_noise = False):
        '''
        Summarizes class balance of data and show a few example images

        Arguments:
            added_noise(boolean):
                Indicates whether to show the data with noise added or not
        '''

        # Show noisy images
        if added_noise:
            for i in range(5):
                for dataset in [self.noisy_train_sample, self.noisy_test_sample]:
                    label = int([dataset[1][i]][0]) 
                    plt.title("Label: " + self.label_names[self.dataset][label])
                    plt.imshow(dataset[0][i], cmap = None if self.dataset == "CIFAR10" else "gray")
                    plt.show()
            return

        # Print class balance
        all_labels = np.concatenate((self.train_sample[1],self.test_sample[1]))
        unique_elem, counts_elem = np.unique(all_labels, return_counts=True)
        print(np.asarray((unique_elem, counts_elem)))

        # Show clean images
        for i in range(5):
            for dataset in [self.train_sample, self.test_sample]:
                label = int([dataset[1][i]][0]) 
                plt.title("Label: " + self.label_names[self.dataset][label])
                plt.imshow(dataset[0][i], cmap = None if self.dataset == "CIFAR10" else "gray")
                plt.show()

    def load_dataset(self, dataset_type, summary = False):
        '''
        Loads a specified dataset with train-test split, and give a summary of
        the dataset is required.

        Arguments:
            dataset_type(string):
                Specifies the type of dataset to be loaded
            summary(bool):
                Specifies whether a data summary will be printed
        '''

        # Load sppropriate dataset with train-test split
        if dataset_type == "MNIST":
            mnist = tf.keras.datasets.mnist.load_data(path='mnist.npz')
            self.train_sample, self.test_sample = mnist[0], mnist[1]
        elif dataset_type == "FMNIST":
            self.train_sample, self.test_sample = tf.keras.datasets.fashion_mnist.load_data()
        elif dataset_type == "CIFAR10":
            self.train_sample, self.test_sample = tf.keras.datasets.cifar10.load_data()
        else: raise ValueError("Dataset "+str(dataset_type)+" not supported.")

        # Turn tuples into arrays for future copying and mutability
        self.train_sample = [np.copy(self.train_sample[0]),np.copy(self.train_sample[1])]
        self.test_sample = [np.copy(self.test_sample[0]),np.copy(self.test_sample[1])]
        self.dataset = dataset_type

        # Special label type conversion due to incompatible type for CIFAR only
        if dataset_type == "CIFAR10":
            self.train_sample[1] = np.asarray([_[0] for _ in self.train_sample[1]])
            self.test_sample[1] = np.asarray([_[0] for _ in self.test_sample[1]])

        # Data Summary
        if summary : self.data_summary()


    def add_noise(self, show_image = False):
        '''
        Add the specified type of noise 

        Arguments:
            noise_type(string):
                Indicates the type of noise to add to images
            show_image(bool):
                Sample noisy images will be printed if and only if True
        '''

        # Make copies of clean dataset to avoid modifying the original
        self.noisy_train_sample = [np.copy(self.train_sample[0]), np.copy(self.train_sample[1])]
        self.noisy_test_sample = [np.copy(self.test_sample[0]), np.copy(self.test_sample[1])]

        # Rotate image randomly by up to +-25 degrees
        def rotate(image, lo, hi): 
            direction = -1 if np.random.randint(10) % 2 == 1 else 1
            angle = np.random.randint(lo, high = hi) * direction
            transformation_matrix = cv2.getRotationMatrix2D(tuple(np.array
                   (image.shape[1::-1]) / 2), angle, 1.0)
            transformed_image = cv2.warpAffine(image, transformation_matrix, 
                                  image.shape[1::-1],flags=cv2.INTER_AREA)
            return transformed_image

        # Randomly select pixels to add Gaussian noise 
        def gaussian(image, variance):
            # normalize to fit the criteria of the function
            image = image / 255.
            image = random_noise(image, mode='gaussian', clip=True, mean=0, var=variance)
            # reverse normalization
            image = image * 255.
            return image

        # Randomly crops rectangles. These crops will be black for grayscale images, or a random 
        # color for colored images.
        def crop(image, num_block):
            im = np.copy(image)
            for block in range(num_block): # Each block has random size
                crop_width, crop_height = np.random.randint(7) + 4, np.random.randint(7) + 4
                if self.dataset == "CIFAR10": # colored image
                    left,top = np.random.randint(31-crop_width), np.random.randint(31-crop_height)
                    random_rgb = np.random.randint(255,size=(3,)) # pick a random color
                    for i in range (top, top + crop_height + 1):
                        for j in range(left, left + crop_width + 1):
                            for k in range (3): im[i][j][k] = random_rgb[k] # fill with random color
                else: # black and white
                    not_blocked = True
                    while not_blocked: # make sure the crop blocks the image
                        left = np.random.randint(27 - crop_width)
                        top = np.random.randint(27 - crop_height)
                        for i in range (top, top + crop_height + 1):
                            for j in range(left, left + crop_width + 1):
                                if im[i][j] != 0: not_blocked = False
                                im[i][j] = 0 # fill the area with black
            return im

        # Add noise to both training and testing sets
        for dataset in [self.noisy_train_sample[0], self.noisy_test_sample[0]]:
            chunk_size = len(dataset) // 10
            for i in range(chunk_size): dataset[i] = gaussian(dataset[i], 0.05)
            for i in range(1 * chunk_size, 2 * chunk_size):dataset[i] = gaussian(dataset[i], 0.08)
            for i in range(2 * chunk_size, 3 * chunk_size):dataset[i] = gaussian(dataset[i], 0.1)
            for i in range(3 * chunk_size, 4 * chunk_size):dataset[i] = rotate(dataset[i], 5, 10)
            for i in range(4 * chunk_size, 5 * chunk_size):dataset[i] = rotate(dataset[i], 10, 20)
            for i in range(5 * chunk_size, 6 * chunk_size):dataset[i] = rotate(dataset[i], 20, 30)
            for i in range(6 * chunk_size, 7 * chunk_size):dataset[i] = crop(dataset[i], 1) 
            for i in range(7 * chunk_size, 8 * chunk_size):dataset[i] = crop(dataset[i], 2) 
            for i in range(8 * chunk_size, 9 * chunk_size):dataset[i] = crop(dataset[i], 3) 

        #Shuffle randomly
        for dataset in [self.noisy_train_sample[0], self.noisy_train_sample[1],
                        self.noisy_test_sample[0], self.noisy_test_sample[1],
                        self.train_sample[0], self.train_sample[1],
                        self.test_sample[0], self.test_sample[1]]:
            np.random.seed(42)
            np.random.shuffle(dataset)

        # Show samples of noisy images       
        if show_image: self.data_summary(True)
    

    def preprocess_data(self):
        '''
        Preprocesses data into appropriate shapes and forms for training and testing
        '''

        # Load training and testing sets
        x_train, x_train_noisy = self.train_sample[0], self.noisy_train_sample[0]
        x_test, x_test_noisy = self.test_sample[0], self.noisy_test_sample[0]
        y_train, y_test = self.train_sample[1], self.test_sample[1] # noisy labels are the same

        # Normalizing input data to 1
        x_train,x_train_noisy = np.true_divide(x_train, 255.0),np.true_divide(x_train_noisy, 255.0) 
        x_test,x_test_noisy = np.true_divide(x_test, 255.0), np.true_divide(x_test_noisy, 255.0)

        # One hot encoding labels
        self.y_train_onehot = np.zeros((y_train.size, 10))
        self.y_train_onehot[np.arange(y_train.size),y_train] = 1
        self.y_test_onehot = np.zeros((y_test.size, 10))
        self.y_test_onehot[np.arange(y_test.size),y_test] = 1

        # Make appropriate shapes of training data
        self.MLP_dim = (3072,) if self.dataset == "CIFAR10" else (784,)
        self.CNN_dim = (32, 32, 3) if self.dataset == "CIFAR10" else (28, 28, 1)
        self.x_train_MLP = x_train.reshape((x_train.shape[0], self.MLP_dim[0]))
        self.x_train_CNN = x_train.reshape((x_train.shape[0], self.CNN_dim[0], self.CNN_dim[1], 
                                            self.CNN_dim[2]))
        self.x_train_noisy_MLP = x_train_noisy.reshape((x_train_noisy.shape[0], self.MLP_dim[0]))
        self.x_train_noisy_CNN = x_train_noisy.reshape((x_train_noisy.shape[0], self.CNN_dim[0],
                                                        self.CNN_dim[1], self.CNN_dim[2]))
        self.x_test_MLP = x_test.reshape((x_test.shape[0], self.MLP_dim[0]))
        self.x_test_CNN = x_test.reshape((x_test.shape[0], self.CNN_dim[0], self.CNN_dim[1], 
                                          self.CNN_dim[2]))
        self.x_test_noisy_MLP = x_test_noisy.reshape((x_test_noisy.shape[0], self.MLP_dim[0]))
        self.x_test_noisy_CNN = x_test_noisy.reshape((x_test_noisy.shape[0], self.CNN_dim[0], 
                                                      self.CNN_dim[1], self.CNN_dim[2]))

        print("Preprocessing Done")


    def train_models(self, graph_type):
        '''
        Train a series of models and show performance graphs. These graphs are for MLP,
        CNN, SDAE and CDAE in noisy and not noisy. Note that graph_type == "Optimizers", "Batch
        Size" and "Learning Rate" are already tested, and best parameters are used in graph_type
        == "Comparison". The first 3 options are left in to allow reproduction of graphs.

        Arguments:
            graph_type(String):
                Specifies the type of hyperparameter for which to produce a graph.
                "Optimizers": Test 7 different optimizers, holding everything else constant.
                "Learning Rate": Test 7 different learning rates, using best optimizer found
                                 in the Optimizers test
                "Batch Size": Test 7 different batch sizes, using best optimizers and learning
                              rates found in previous 2 tests
                "Comparison": Test MLP, CNN, SDAE+MLP, SDAE+CNN, CDAE+MLP, CDAE+CNN on noisy and
                              noiseless dataset, using best hyperparameters found in previous 3
                              tests
        '''
        # Make tick numbers larger
        plt.rc('xtick',labelsize=18)
        plt.rc('ytick',labelsize=18)

        # Define all models for plotting
        model_type = ["MLP-Noisy", "MLP-Noiseless","CNN-Noisy", "CNN-Noiseless",
                      "SDAE-Noisy", "SDAE-Noiseless","CDAE-Noisy", "CDAE-Noiseless"]
        num_epochs = 100

        # Plotting different optimizers
        if graph_type == "Optimizers":
            optimizer_type = [optimizers.SGD(lr = 0.05, nesterov=False), 
                              optimizers.SGD(lr = 0.05, nesterov=True, momentum = 0.95), 
                              optimizers.Adagrad(lr = 0.005), 
                              optimizers.Adadelta(lr = 0.005), optimizers.RMSprop(), 
                              optimizers.Adam(), optimizers.Adamax()]
            results_optm = []
            
            # Run all 4 models under 2 noise types for 100 epochs
            for optm in optimizer_type:
                print(optm)
                results_single = []
                self.build_NN("MLP", self.x_train_noisy_MLP, self.y_train_onehot, self.MLP_dim, 
                              optm, 256, num_epochs, 0)
                results_single.append(self.history)
                results = self.classifier.evaluate(self.x_test_noisy_MLP, self.y_test_onehot)
                print("\n"+str(optm) + " MLP noisy accuracy: "+str(results[1]))
                self.build_NN("MLP", self.x_train_MLP, self.y_train_onehot, self.MLP_dim, 
                              optm, 256, num_epochs, 0)
                results_single.append(self.history)
                results = self.classifier.evaluate(self.x_test_MLP, self.y_test_onehot)
                print("\n"+str(optm) + " MLP noiseless accuracy: "+str(results[1]))
                self.build_NN("CNN", self.x_train_noisy_CNN, self.y_train_onehot, self.CNN_dim, 
                              optm, 256, num_epochs, 0)
                results_single.append(self.history)
                results = self.classifier.evaluate(self.x_test_noisy_CNN, self.y_test_onehot)
                print("\n"+str(optm) + " CNN noisy accuracy: "+str(results[1]))
                self.build_NN("CNN", self.x_train_CNN, self.y_train_onehot, self.CNN_dim, 
                              optm, 256, num_epochs, 0)
                results_single.append(self.history)
                results = self.classifier.evaluate(self.x_test_CNN, self.y_test_onehot)
                print("\n"+str(optm) + " CNN noiseless accuracy: "+str(results[1]))
                self.build_NN("SDAE", self.x_train_noisy_MLP, self.x_train_MLP, self.MLP_dim, 
                              optm, 256, num_epochs, 0)
                results_single.append(self.history)
                results = self.denoiser.evaluate(self.x_test_noisy_MLP, self.x_test_MLP)
                print("\n"+str(optm) + " SDAE noisy Loss: "+str(results[0]))
                self.build_NN("SDAE", self.x_train_MLP, self.x_train_MLP, self.MLP_dim, 
                              optm, 256, num_epochs, 0)
                results_single.append(self.history)
                results = self.denoiser.evaluate(self.x_test_MLP, self.x_test_MLP)
                print("\n"+str(optm) + " SDAE noiseless Loss: "+str(results[0]))
                self.build_NN("CDAE", self.x_train_noisy_CNN, self.x_train_CNN, self.CNN_dim, 
                              optm, 256, num_epochs, 0)
                results_single.append(self.history)
                results = self.denoiser.evaluate(self.x_test_noisy_CNN, self.x_test_CNN)
                print("\n"+str(optm) + " CDAE noisy Loss: "+str(results[0]))
                self.build_NN("CDAE", self.x_train_CNN, self.x_train_CNN, self.CNN_dim, 
                              optm, 256, num_epochs, 0)
                results_single.append(self.history)
                results = self.denoiser.evaluate(self.x_test_CNN, self.x_test_CNN)
                print("\n"+str(optm) + " CDAE noiseless Loss: "+str(results[0]))
                results_optm.append(results_single)
                print(results_optm)

            # Reshape data from per-optimizer to per-model
            to_plot = [[_[i] for _ in results_optm] for i in range(8)]

            # Define appropriate ranges and labels
            x_ticks = list(range(0, 101, 10))
            if self.dataset == "MNIST":  y_ranges = [(0.9, 1)] *4 + [(0, 0.04)] *4
            elif self.dataset == "FMNIST":  y_ranges = [(0.7, 0.95)] *4 +[(0, 0.05)] *4
            elif self.dataset == "CIFAR10":  y_ranges = [(0.3, 0.88)] *4 +[(0, 0.04)] *4

            # Plot 8 graphs for 8 models
            for i in range(8):
                fig, ax = plt.subplots(figsize = (10, 7))
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
                #plt.title(self.dataset+" "+model_type[i], fontsize = 20)
                plt.ylim(y_ranges[i])
                plt.xticks(x_ticks)
                #plt.ylabel("Accuracy" if i < 4 else "Loss (MSE)", fontsize = 18)
                #plt.xlabel("Epoch Number", fontsize = 18)
                ax.plot(list(range(1, num_epochs+1)), to_plot[i][0], label="Gradient Descent", )
                ax.plot(list(range(1, num_epochs+1)), to_plot[i][1], label="GD + Nesterov momentum", )
                ax.plot(list(range(1, num_epochs+1)), to_plot[i][2], label="Adagrad", )
                ax.plot(list(range(1, num_epochs+1)), to_plot[i][3], label="Adadelta",)
                ax.plot(list(range(1, num_epochs+1)), to_plot[i][4], label="RMSProp", )
                ax.plot(list(range(1, num_epochs+1)), to_plot[i][5], label="Adam", )
                ax.plot(list(range(1, num_epochs+1)), to_plot[i][6], label="Adamax", )
                #plt.legend(loc="best", fontsize = 18)

        # Plotting different learning rates
        elif graph_type == "Learning Rate":
            results_LR = []
            learning_rates = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1]

            # Run all 4 models under 2 noise types for 100 epochs
            for LR in learning_rates:

                # Each test uses the best optimizer found in the Optimizers test
                print(LR)
                results_single = []
                self.build_NN("MLP", self.x_train_noisy_MLP, self.y_train_onehot, self.MLP_dim, 
                              optimizers.Adamax(LR), 256, num_epochs, 0)
                results_single.append(self.history)
                results = self.classifier.evaluate(self.x_test_noisy_MLP, self.y_test_onehot)
                print("\n"+str(LR) + " MLP noisy accuracy: "+str(results[1]))
                self.build_NN("MLP", self.x_train_MLP, self.y_train_onehot, self.MLP_dim, 
                              optimizers.Adamax(LR), 256, num_epochs, 0)
                results_single.append(self.history)
                results = self.classifier.evaluate(self.x_test_MLP, self.y_test_onehot)
                print("\n"+str(LR) + " MLP noiseless accuracy: "+str(results[1]))
                self.build_NN("CNN", self.x_train_noisy_CNN, self.y_train_onehot, self.CNN_dim, 
                              optimizers.Adamax(LR), 256, num_epochs, 0)
                results_single.append(self.history)
                results = self.classifier.evaluate(self.x_test_noisy_CNN, self.y_test_onehot)
                print("\n"+str(LR) + " CNN noisy accuracy: "+str(results[1]))
                self.build_NN("CNN", self.x_train_CNN, self.y_train_onehot, self.CNN_dim, 
                              optimizers.Adamax(LR), 256, num_epochs, 0)
                results_single.append(self.history)
                results = self.classifier.evaluate(self.x_test_CNN, self.y_test_onehot)
                print("\n"+str(LR) + " CNN noiseless accuracy: "+str(results[1]))
                self.build_NN("SDAE", self.x_train_noisy_MLP, self.x_train_MLP, self.MLP_dim, 
                              optimizers.Adamax(LR), 256, num_epochs, 0)
                results_single.append(self.history)
                results = self.denoiser.evaluate(self.x_test_noisy_MLP, self.x_test_MLP)
                print("\n"+str(LR) + " SDAE noisy Loss: "+str(results[0]))
                self.build_NN("SDAE", self.x_train_MLP, self.x_train_MLP, self.MLP_dim, 
                              optimizers.Adamax(LR), 256, num_epochs, 0)
                results_single.append(self.history)
                results = self.denoiser.evaluate(self.x_test_MLP, self.x_test_MLP)
                print("\n"+str(LR) + " SDAE noiseless Loss: "+str(results[0]))
                self.build_NN("CDAE", self.x_train_noisy_CNN, self.x_train_CNN, self.CNN_dim, 
                              optimizers.Adamax(LR), 256, num_epochs, 0)
                results_single.append(self.history)
                results = self.denoiser.evaluate(self.x_test_noisy_CNN, self.x_test_CNN)
                print("\n"+str(LR) + " CDAE noisy Loss: "+str(results[0]))
                self.build_NN("CDAE", self.x_train_CNN, self.x_train_CNN, self.CNN_dim, 
                              optimizers.Adamax(LR), 256, num_epochs, 0)
                results_single.append(self.history)
                results = self.denoiser.evaluate(self.x_test_CNN, self.x_test_CNN)
                print("\n"+str(LR) + " CDAE noiseless Loss: "+str(results[0]))
                results_LR.append(results_single)
                print(results_LR)

            # Reshape data from per-optimizer to per-model
            to_plot = [[_[i] for _ in results_LR] for i in range(8)]

            # Define appropriate ranges and labels
            x_ticks = list(range(0, 101, 10))
            if self.dataset == "MNIST": y_ranges = [(0.92, 1)] *4 + [(0, 0.025)] *4
            elif self.dataset == "FMNIST": y_ranges = [(0.75, 0.95)] *4 +[(0, 0.03)] *4
            elif self.dataset == "CIFAR10": y_ranges = [(0.2, 0.9)] *4 +[(0, 0.03)] *4

            # Plot 8 graphs for 8 models
            for i in range(8):
                fig, ax = plt.subplots(figsize = (10, 7))
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
                #plt.title(self.dataset+" "+model_type[i], fontsize = 20)
                plt.ylim(y_ranges[i])
                plt.xticks(x_ticks)
                #plt.ylabel("Accuracy" if i < 4 else "Loss (MSE)", fontsize = 18)
                #plt.xlabel("Epoch Number", fontsize = 18)
                ax.plot(list(range(1, num_epochs+1)), to_plot[i][0], label="LR = 0.0001", )
                ax.plot(list(range(1, num_epochs+1)), to_plot[i][1], label="LR = 0.0003", )
                ax.plot(list(range(1, num_epochs+1)), to_plot[i][2], label="LR = 0.001", )
                ax.plot(list(range(1, num_epochs+1)), to_plot[i][3], label="LR = 0.003",)
                ax.plot(list(range(1, num_epochs+1)), to_plot[i][4], label="LR = 0.01", )
                ax.plot(list(range(1, num_epochs+1)), to_plot[i][5], label="LR = 0.03", )
                ax.plot(list(range(1, num_epochs+1)), to_plot[i][6], label="LR = 0.1", )
                #plt.legend(loc="best", fontsize = 18)

        # Plotting different batch size
        elif graph_type == "Batch Size":
            results_BS = []
            batch_sizes = [1024, 512, 256, 128, 64, 32, 16]

            # Run all 4 models under 2 noise types for 100 epochs
            for BS in batch_sizes:

                # Each test uses the best optimizer found in the Optimizers test and the best LR
                # found in the LR test
                print(BS)
                results_single = []
                self.build_NN("MLP", self.x_train_noisy_MLP, self.y_train_onehot, self.MLP_dim, 
                              optimizers.Adamax(0.003), BS, num_epochs, 0)
                results = self.classifier.evaluate(self.x_test_noisy_MLP, self.y_test_onehot)
                print("\n"+str(BS) + " MLP noisy accuracy: "+str(results[1]))
                results_single.append(self.history)
                self.build_NN("MLP", self.x_train_MLP, self.y_train_onehot, self.MLP_dim, 
                              optimizers.Adamax(0.003), BS, num_epochs, 0)
                results = self.classifier.evaluate(self.x_test_MLP, self.y_test_onehot)
                print("\n"+str(BS) + " MLP noiseless accuracy: "+str(results[1]))
                results_single.append(self.history)
                self.build_NN("CNN", self.x_train_noisy_CNN, self.y_train_onehot, self.CNN_dim, 
                              optimizers.Adamax(0.001), BS, num_epochs, 0)
                results = self.classifier.evaluate(self.x_test_noisy_CNN, self.y_test_onehot)
                print("\n"+str(BS) + " CNN noisy accuracy: "+str(results[1]))
                results_single.append(self.history)
                self.build_NN("CNN", self.x_train_CNN, self.y_train_onehot, self.CNN_dim, 
                              optimizers.Adamax(0.001), BS, num_epochs, 0)
                results = self.classifier.evaluate(self.x_test_CNN, self.y_test_onehot)
                print("\n"+str(BS) + " CNN noiseless accuracy: "+str(results[1]))
                results_single.append(self.history)
                self.build_NN("SDAE", self.x_train_noisy_MLP, self.x_train_MLP, self.MLP_dim, 
                              optimizers.Adamax(0.003), BS, num_epochs, 0)
                results = self.denoiser.evaluate(self.x_test_noisy_MLP, self.x_test_MLP)
                print("\n"+str(BS) + " SDAE noisy Loss: "+str(results[0]))
                results_single.append(self.history)
                self.build_NN("SDAE", self.x_train_MLP, self.x_train_MLP, self.MLP_dim, 
                              optimizers.Adamax(0.003), BS, num_epochs, 0)
                results = self.denoiser.evaluate(self.x_test_MLP, self.x_test_MLP)
                print("\n"+str(BS) + " SDAE noiseless Loss: "+str(results[0]))
                results_single.append(self.history)
                self.build_NN("CDAE", self.x_train_noisy_CNN, self.x_train_CNN, self.CNN_dim, 
                              optimizers.Adamax(0.003), BS, num_epochs, 0)
                results = self.denoiser.evaluate(self.x_test_noisy_CNN, self.x_test_CNN)
                print("\n"+str(BS) + " CDAE noisy Loss: "+str(results[0]))
                results_single.append(self.history)
                self.build_NN("CDAE", self.x_train_CNN, self.x_train_CNN, self.CNN_dim, 
                              optimizers.Adamax(0.003), BS, num_epochs, 0)
                results = self.denoiser.evaluate(self.x_test_CNN, self.x_test_CNN)
                print("\n"+str(BS) + " CDAE noiseless Loss: "+str(results[0]))
                results_single.append(self.history)
                results_BS.append(results_single)
                print(results_BS)

            # Reshape data from per-optimizer to per-model
            to_plot = [[_[i] for _ in results_BS] for i in range(8)]
            print(to_plot)

            # Define appropriate ranges and labels
            x_ticks = list(range(0, 101, 10))
            if self.dataset == "MNIST": y_ranges = [(0.92, 1)] *4 + [(0, 0.025)] *4
            elif self.dataset == "FMNIST": y_ranges = [(0.8, 0.95)] *4 +[(0, 0.02)] *4
            elif self.dataset == "CIFAR10": y_ranges = [(0.4, 0.88)] *4 +[(0, 0.022)] *4

            # Plot 8 graphs for 8 models
            for i in range(8):
                fig, ax = plt.subplots(figsize = (10, 7))
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
                #plt.title(self.dataset+" "+model_type[i], fontsize = 20)
                plt.ylim(y_ranges[i])
                plt.xticks(x_ticks)
                #plt.ylabel("Accuracy" if i < 4 else "Loss (MSE)", fontsize = 18)
                #plt.xlabel("Epoch Number", fontsize = 18)
                ax.plot(list(range(1, num_epochs+1)), to_plot[i][0], label="BS = 1024", )
                ax.plot(list(range(1, num_epochs+1)), to_plot[i][1], label="BS = 512", )
                ax.plot(list(range(1, num_epochs+1)), to_plot[i][2], label="BS = 256", )
                ax.plot(list(range(1, num_epochs+1)), to_plot[i][3], label="BS = 128",)
                ax.plot(list(range(1, num_epochs+1)), to_plot[i][4], label="BS = 64", )
                ax.plot(list(range(1, num_epochs+1)), to_plot[i][5], label="BS = 32", )
                ax.plot(list(range(1, num_epochs+1)), to_plot[i][6], label="BS = 16", )
                #plt.legend(loc="best", fontsize = 18)

        # Final results compasion, all models are trained with with optimal Optimizer, 
        # LR and BS found above
        elif graph_type == "Comparison":
        
            # Set up some variables
            num_epochs_special = 40
            noiseless_results = []
            noisy_results = []
            SDAE_denoised_noiseless_train, SDAE_denoised_noiseless_test = None, None
            CDAE_denoised_noiseless_train, CDAE_denoised_noiseless_test = None, None
            SDAE_denoised_noisy_train, SDAE_denoised_noisy_test = None, None
            CDAE_denoised_noisy_train, CDAE_denoised_noisy_test = None, None
            
            # First denoise both noiseless and noisy sets with SDAE and CDAE
            self.build_NN("SDAE", self.x_train_MLP, self.x_train_MLP, self.MLP_dim, 
                          optimizers.Adamax(0.003), 32, num_epochs, 0)
            SDAE_denoised_noiseless_train = self.denoiser.predict(self.x_train_MLP)
            SDAE_denoised_noiseless_test = self.denoiser.predict(self.x_test_MLP)
            self.build_NN("SDAE", self.x_train_noisy_MLP, self.x_train_MLP, self.MLP_dim, 
                          optimizers.Adamax(0.003), 32, num_epochs, 0)
            SDAE_denoised_noisy_train = self.denoiser.predict(self.x_train_noisy_MLP)
            SDAE_denoised_noisy_test = self.denoiser.predict(self.x_test_noisy_MLP)
            self.build_NN("CDAE", self.x_train_CNN, self.x_train_CNN, self.CNN_dim, 
                           optimizers.Adamax(0.001), 64, num_epochs_special, 0) #Special:40 epoch instead of 100
            CDAE_denoised_noiseless_train = self.denoiser.predict(self.x_train_CNN)
            CDAE_denoised_noiseless_test = self.denoiser.predict(self.x_test_CNN)
            self.build_NN("CDAE", self.x_train_noisy_CNN, self.x_train_CNN, self.CNN_dim, 
                           optimizers.Adamax(0.001), 64, num_epochs_special, 0) #Special:40 epoch instead of 100
            CDAE_denoised_noisy_train = self.denoiser.predict(self.x_train_noisy_CNN)
            CDAE_denoised_noisy_test = self.denoiser.predict(self.x_test_noisy_CNN)

            # Specify some dataset grouping for upcoming training
            datasets = [[SDAE_denoised_noiseless_train, SDAE_denoised_noiseless_test],
                        [SDAE_denoised_noisy_train, SDAE_denoised_noisy_test],
                        [CDAE_denoised_noiseless_train, CDAE_denoised_noiseless_test],
                        [CDAE_denoised_noisy_train, CDAE_denoised_noisy_test],]
            MLP_names = ["SDAE+MLP_SDAE Noiseless", "SDAE+MLP_SDAE Noisy",
                         "CDAE+MLP_CDAE Noiseless", "CDAE+MLP_CDAE Noisy",]
            MLP_names2 = ["SDAE+MLP_Noiseless Noiseless", "SDAE+MLP_Noiseless Noisy",
                          "CDAE+MLP_Noiseless Noiseless", "CDAE+MLP_Noiseless Noisy",]
            CNN_names = ["SDAE+CNN_SDAE Noiseless", "SDAE+CNN_SDAE Noisy",
                         "CDAE+CNN_CDAE Noiseless", "CDAE+CNN_CDAE Noisy",]
            CNN_names2 = ["SDAE+CNN_Noiseless Noiseless", "SDAE+CNN_Noiseless Noisy",
                          "CDAE+CNN_Noiseless Noiseless", "CDAE+CNN_Noiseless Noisy",]

            # Train MLP on noiseless data
            self.build_NN("MLP", self.x_train_MLP, self.y_train_onehot, self.MLP_dim, 
                          optimizers.Adamax(0.003), 64, num_epochs, 0)
            noiseless_results.append(self.history)

            # Test noiseless-trained MLP on noiseless data, "denoised" noiseless data and 
            # denoised noisy data
            results = self.classifier.evaluate(self.x_test_MLP, self.y_test_onehot)
            print("MLP_Noiseless Noiseless\nLoss: "+str(results[0])+"\nAccuracy: "+str(results[1]))
            for dataset, MLP_name in zip(datasets, MLP_names2):
                #Pass in a separate validation set
                denoised_train = dataset[0].reshape((dataset[0].shape[0], self.MLP_dim[0]))
                self.build_NN("MLP", self.x_train_MLP, self.y_train_onehot, self.MLP_dim, 
                          optimizers.Adamax(0.003), 64, num_epochs, 0, denoised_train)
                if "Noisy" in MLP_name: noisy_results.append(self.history)
                else: noiseless_results.append(self.history)
                denoised_test = dataset[1].reshape((dataset[1].shape[0], self.MLP_dim[0]))
                results = self.classifier.evaluate(denoised_test, self.y_test_onehot)
                print(MLP_name+"\nLoss: "+str(results[0])+"\nAccuracy: "+str(results[1]))

            # Train CNN on noiseless data.
            self.build_NN("CNN", self.x_train_CNN, self.y_train_onehot, self.CNN_dim, 
                          optimizers.Adamax(0.001), 16, num_epochs, 0)
            noiseless_results.append(self.history)

            # Test noiseless-trained CNN on noiseless data, "denoised" noiseless data and 
            # denoised noisy data
            results = self.classifier.evaluate(self.x_test_CNN, self.y_test_onehot)
            print("CNN_Noiseless Noiseless\nLoss: "+str(results[0])+"\nAccuracy: "+str(results[1]))
            for dataset, CNN_name in zip(datasets, CNN_names2):
                denoised_train = dataset[0].reshape((dataset[0].shape[0], self.CNN_dim[0], 
                                                    self.CNN_dim[1], self.CNN_dim[2]))
                self.build_NN("CNN", self.x_train_CNN, self.y_train_onehot, self.CNN_dim, 
                          optimizers.Adamax(0.001), 16, num_epochs, 0, denoised_train)
                if "Noisy" in CNN_name: noisy_results.append(self.history)
                else: noiseless_results.append(self.history)
                denoised_test = dataset[1].reshape((dataset[1].shape[0], self.CNN_dim[0], 
                                                    self.CNN_dim[1], self.CNN_dim[2]))
                results = self.classifier.evaluate(denoised_test, self.y_test_onehot)
                print(CNN_name + "\nLoss: "+str(results[0])+"\nAccuracy: "+str(results[1]))

            # Train and test MLP on noisy data
            self.build_NN("MLP", self.x_train_noisy_MLP, self.y_train_onehot, self.MLP_dim, 
                          optimizers.Adamax(0.003), 64, num_epochs, 0)
            noisy_results.append(self.history)
            results = self.classifier.evaluate(self.x_test_noisy_MLP, self.y_test_onehot)
            print("MLP_Noisy Noisy\nLoss: "+str(results[0])+"\nAccuracy: "+str(results[1]))

            # Train and test MLP on SDAE- and CDAE- denoised noiseless and noisy data
            for dataset, MLP_name in zip(datasets, MLP_names):
                denoised_train = dataset[0].reshape((dataset[0].shape[0], self.MLP_dim[0]))
                denoised_test = dataset[1].reshape((dataset[1].shape[0], self.MLP_dim[0]))
                self.build_NN("MLP", denoised_train, self.y_train_onehot, self.MLP_dim, 
                            optimizers.Adamax(0.003), 64, num_epochs, 0)
                if "Noisy" in MLP_name: noisy_results.append(self.history)
                else: noiseless_results.append(self.history)
                results = self.classifier.evaluate(denoised_test, self.y_test_onehot)
                print(MLP_name+"\nLoss: "+str(results[0])+"\nAccuracy: "+str(results[1]))

            # Train and test CNN on noisy data
            self.build_NN("CNN", self.x_train_noisy_CNN, self.y_train_onehot, self.CNN_dim, 
                          optimizers.Adamax(0.001), 16, num_epochs, 0)
            noisy_results.append(self.history)
            results = self.classifier.evaluate(self.x_test_noisy_CNN, self.y_test_onehot)
            print("CNN_Noisy Noisy\nLoss: "+str(results[0])+"\nAccuracy: "+str(results[1]))

            # Train and test MLP on SDAE- and CDAE- denoised noiseless and noisy data
            for dataset, CNN_name in zip(datasets, CNN_names):
                denoised_train = dataset[0].reshape((dataset[0].shape[0], self.CNN_dim[0], 
                                                     self.CNN_dim[1], self.CNN_dim[2]))
                denoised_test = dataset[1].reshape((dataset[1].shape[0], self.CNN_dim[0], 
                                                    self.CNN_dim[1], self.CNN_dim[2]))
                self.build_NN("CNN", denoised_train, self.y_train_onehot, self.CNN_dim, 
                              optimizers.Adamax(0.001), 16, num_epochs, 0)
                if "Noisy" in CNN_name: noisy_results.append(self.history)
                else: noiseless_results.append(self.history)
                results = self.classifier.evaluate(denoised_test, self.y_test_onehot)
                print(CNN_name + "\nLoss: "+str(results[0])+"\nAccuracy: "+str(results[1]))

            #print(noisy_results)
            #print(noiseless_results)

    def build_NN(self, model, x_train, y_train, input_dim, optm, batch_sz, epochs, verbose, val_data = None):
        '''
        Build and train a classifier model

        Arguments:
            model(string):
                Specifies the type of classifier network, "MLP" or "CNN"
            x_train(numpy.ndarray):
                Aarrays representing the training image
            y_train(numpy.ndarray):
                Labels corresponding to the training images
            input_dim(tuple):
                Specifies the dimension of the input to the neural network
            optm(keras.optimizer or any compatible class):
                The optimizer for the neural network
            batch_sz(integer):
                The number of samples in a minibatch
            verbose(int):
                How much info the model will provide during training
            val_data(numpy.ndarray):
                Allows passing in different data than x_train for validation
        '''
        validation_cutoff = int(0.9 * int(x_train.shape[0]))

        # "Force" the last chunk of x_train to use the new validation data, if given any.
        if val_data is not None:
            if x_train.shape != val_data.shape:
                raise ValueError("x_train must have same shape as validation data")
            x_train = np.concatenate((x_train[:validation_cutoff], val_data[validation_cutoff:]))

        # MLP model: Input (784 for MNIST and Fashion MNIST, or 3072 for CIFAR-10) - [Dense (1024) 
        # - Elu - Dropout (0.2)] x 3  - Dense (10) - Softmax - Output (1)
        if model == "MLP":
            model = Sequential()
            model.add(Dense(1024, input_shape = input_dim))
            model.add(Activation('elu'))
            model.add(Dropout(0.2))
            for i in range(2):
                model.add(Dense(1024))
                model.add(Activation('elu'))
                model.add(Dropout(0.2))
            model.add(Dense(10))
            model.add(Activation('softmax'))
            #print(model.summary())
            model.compile(optimizer=optm, loss='categorical_crossentropy', metrics=['accuracy'])
            history = model.fit(x_train[:validation_cutoff], y_train[:validation_cutoff], 
                                batch_size=batch_sz, epochs=epochs, verbose=verbose, 
                                validation_data=(x_train[validation_cutoff:],y_train[validation_cutoff:]))
            self.classifier = model
            self.history = history.history['val_accuracy']
        
        # CNN model: Input (28x28 for MNIST and Fashion MNIST, or 32x32x3 for CIFAR-10) - 
        # [2DConv (3x3, 32 windows) - Elu - BatchNorm] x 2 - Maxpool (2x2) - Dropout (0.2) - 
        # [2DConv (3x3, 64 windows) - Elu - BatchNorm] x 2 - Maxpool (2x2) - Dropout (0.3) - 
        # [2DConv (3x3, 128 windows) - Elu - BatchNorm] x 2 - Maxpool (2x2) - Dropout (0.4) - 
        # Flatten - Dense (10) - Softmax - Output (1)
        elif model == "CNN":
            model = Sequential()
            model.add(Conv2D(32, (3,3), padding='same', 
                             kernel_regularizer=regularizers.l2(0.0001), 
                             input_shape=input_dim))
            model.add(Activation('elu'))
            model.add(BatchNormalization())
            model.add(Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizers.l2(0.0001)))
            model.add(Activation('elu'))
            model.add(BatchNormalization())
            model.add(MaxPooling2D(pool_size=(2,2)))
            model.add(Dropout(0.2))
            for i in range(2):
                model.add(Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizers.l2(0.0001)))
                model.add(Activation('elu'))
                model.add(BatchNormalization())
            model.add(MaxPooling2D(pool_size=(2,2)))
            model.add(Dropout(0.3))
            for i in range(2):
                model.add(Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizers.l2(0.0001)))
                model.add(Activation('elu'))
                model.add(BatchNormalization())
            model.add(MaxPooling2D(pool_size=(2,2)))
            model.add(Dropout(0.4))
            model.add(Flatten())
            model.add(Dense(10, activation='softmax'))
            #print(model.summary())
            model.compile(optimizer=optm, loss='categorical_crossentropy', metrics=['accuracy'])
            history = model.fit(x_train[:validation_cutoff], y_train[:validation_cutoff], 
                                batch_size=batch_sz, epochs=epochs, verbose=verbose, 
                                validation_data=(x_train[validation_cutoff:],y_train[validation_cutoff:]))
            self.classifier = model
            self.history = history.history['val_accuracy']

        # SDAE model: input (784 for MNIST and Fashion MNIST, or 3072 for CIFAR-10) - Dense (512) 
        # - Elu - Dropout (0.2) - Dense (128) - Elu - Dropout (0.2) - Dense (512) - Elu - 
        # Dropout (0.2) - Dense (same shape as input) - Sigmoid - Output (same shape as input)
        elif model == "SDAE":
            model = Sequential()
            model.add(Dense(512, input_shape = input_dim))
            model.add(Activation('elu'))
            model.add(Dropout(0.2))
            model.add(Dense(128))
            model.add(Activation('elu'))
            model.add(Dropout(0.2))
            model.add(Dense(512))
            model.add(Activation('elu'))
            model.add(Dropout(0.2))
            model.add(Dense(input_dim[0]))
            model.add(Activation("sigmoid"))
            #print(model.summary())
            model.compile(optimizer=optm, loss="mse", metrics=["mse"])
            history = model.fit(x_train[:validation_cutoff], y_train[:validation_cutoff], 
                                batch_size=batch_sz, epochs=epochs, verbose=verbose, 
                                validation_data=(x_train[validation_cutoff:],y_train[validation_cutoff:]))
            self.denoiser = model
            self.history = history.history['val_loss']
        
        # Define Block(a, b) as: [2DConv(3x3, a windows) - Elu - BatchNorm] x b

        # CDAE model: Input (28x28 for MNIST and Fashion MNIST, or 32x32x3 for CIFAR-10) 
        # - Block (32, 2) - Maxpool (2x2) - Dropout (0.2) - Block (64, 2) - Maxpool (2x2) - 
        # Dropout (0.3) - Block (128, 2) - Maxpool (2x2) - Dropout (0.4) - Block (256, 4) - 
        # Block (128, 2) - Upsample (2x2) - Block (64, 2) - Block (32, 2) - Upsample (2x2) - 
        # 2DConv (3x3, 1 window for MNIST or Fashion MNIST, or 3 windows for CIFAR-10) - 
        # Sigmoid - Output (same shape as input)
        elif model == "CDAE":
            model = Sequential()
            model.add(Input(shape=input_dim))
            for i in range(2):
                model.add(Conv2D(32, (3, 3), activation='elu', padding='same'))
                model.add(BatchNormalization())
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.2))
            for i in range(2):
                model.add(Conv2D(64, (3, 3), activation='elu', padding='same'))
                model.add(BatchNormalization())
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.3))
            for i in range(2):
                model.add(Conv2D(128, (3, 3), activation='elu', padding='same'))
                model.add(BatchNormalization())
            model.add(Dropout(0.4))
            for i in range(4):
                model.add(Conv2D(256, (3, 3), activation='elu', padding='same'))
                model.add(BatchNormalization())
            for i in range(2):
                model.add(Conv2D(128, (3, 3), activation='elu', padding='same'))
                model.add(BatchNormalization())
            model.add(UpSampling2D((2,2)))
            for i in range(2):
                model.add(Conv2D(64, (3, 3), activation='elu', padding='same'))
                model.add(BatchNormalization())
            for i in range(2):
                model.add(Conv2D(32, (3, 3), activation='elu', padding='same'))
                model.add(BatchNormalization())
            model.add(UpSampling2D((2,2)))
            model.add(Conv2D(input_dim[-1], (3, 3), activation='sigmoid', padding='same'))
            #print(model.summary())
            model.compile(optimizer=optm, loss='mse', metrics=["mse"])
            history = model.fit(x_train[:validation_cutoff], y_train[:validation_cutoff], 
                                batch_size=batch_sz, epochs=epochs, verbose=verbose, 
                                validation_data=(x_train[validation_cutoff:],y_train[validation_cutoff:]))
            self.denoiser = model
            self.history = history.history['val_loss']
    

    def tsne(self):
        '''
            Select 10% of data randomly from training and testing set (because full set will take
            too much time) to perform TSNE visualization
        '''

        for datasets, desc in zip([(self.x_train_MLP, self.x_test_MLP), 
                                   (self.x_train_noisy_MLP, self.x_test_noisy_MLP)],
                                  ["Noiseless", "Noisy"]):
            # Combine training and testing samples, and training and testing labels
            image_pixels = pd.DataFrame(np.concatenate((datasets[0], datasets[1])))
            labels_concat = np.concatenate((self.y_train_onehot, self.y_test_onehot))
            labels_flattened = np.asarray([np.where(x == 1)[0][0] for x in labels_concat])
            # Converting to pd to fit criteria of the TSNE function
            image_labels = pd.DataFrame(labels_flattened)
            # Random selection
            image_pixels = image_pixels.sample(frac=0.1, random_state=42).reset_index(drop=True)
            image_labels = image_labels.sample(frac=0.1, random_state=42).reset_index(drop=True)  
            df = image_pixels
            # Do TSNE
            tsne = TSNE(n_iter=700)
            tsne_results = tsne.fit_transform(df.values)
            # Add Labels for plotting differentiation of classes
            df['label'] = image_labels
            plt.figure(figsize=(9,9))
            plt.title("TSNE Plot on 10% of Examples Randomly Sampled From " + self.dataset+", "+desc)
            plt.xlabel("TSNE Reduced Mapping Value, Dimension 1")
            plt.ylabel("TSNE Reduced Mapping Value, Dimension 2")
            
            # Grab datapoints for each class and plot it with appropriate label
            for i in range(10):
                label_name = self.label_names[self.dataset][i]
                idx = np.asarray(df.index[df['label'] == i].tolist())
                plt.scatter(x=tsne_results[:,0][idx], y=tsne_results[:,1][idx], alpha=0.35,
                            label = label_name)
            plt.legend(loc="best")
            plt.show()
    
    def show_denoised_images(self):
        '''
        Prints sample results of "denoising" a noiseless image and denoising a noisy image, using
        the SDAE and CDAE
        '''
        # Set up some variables
        num_epochs = 100
        num_epochs_special = 40
        sample_size = 10
        SDAE_denoised_noiseless_test, CDAE_denoised_noiseless_test = None, None
        SDAE_denoised_noisy_test, CDAE_denoised_noisy_test = None, None
        
        # Train denoisers and test on the first sample_size images of test set
        self.build_NN("SDAE", self.x_train_MLP, self.x_train_MLP, self.MLP_dim, 
                        optimizers.Adamax(0.003), 32, num_epochs, 0)
        SDAE_denoised_noiseless_test = self.denoiser.predict(self.x_test_MLP[:sample_size])
        self.build_NN("SDAE", self.x_train_noisy_MLP, self.x_train_MLP, self.MLP_dim, 
                        optimizers.Adamax(0.003), 32, num_epochs, 0)
        SDAE_denoised_noisy_test = self.denoiser.predict(self.x_test_noisy_MLP[:sample_size])
        self.build_NN("CDAE", self.x_train_CNN, self.x_train_CNN, self.CNN_dim, 
                        optimizers.Adamax(0.001), 64, num_epochs_special, 0) #Special:40 epoch instead of 100
        CDAE_denoised_noiseless_test = self.denoiser.predict(self.x_test_CNN[:sample_size])
        self.build_NN("CDAE", self.x_train_noisy_CNN, self.x_train_CNN, self.CNN_dim, 
                        optimizers.Adamax(0.001), 64, num_epochs_special, 0) #Special:40 epoch instead of 100
        CDAE_denoised_noisy_test = self.denoiser.predict(self.x_test_noisy_CNN[:sample_size])
        
        # Reshape into format suitable for showing image
        if self.dataset == "CIFAR10":
            SDAE_denoised_noiseless_test = SDAE_denoised_noiseless_test.reshape((sample_size, 32, 32, 3))
            CDAE_denoised_noiseless_test = CDAE_denoised_noiseless_test.reshape((sample_size, 32, 32, 3))
            SDAE_denoised_noisy_test = SDAE_denoised_noisy_test.reshape((sample_size, 32, 32, 3))
            CDAE_denoised_noisy_test = CDAE_denoised_noisy_test.reshape((sample_size, 32, 32, 3))
        else:
            SDAE_denoised_noiseless_test = SDAE_denoised_noiseless_test.reshape((sample_size, 28, 28))
            CDAE_denoised_noiseless_test = CDAE_denoised_noiseless_test.reshape((sample_size, 28, 28))
            SDAE_denoised_noisy_test = SDAE_denoised_noisy_test.reshape((sample_size, 28, 28))
            CDAE_denoised_noisy_test = CDAE_denoised_noisy_test.reshape((sample_size, 28, 28))

        # Define labels
        labels = self.test_sample[1][:sample_size]

        # Print images 
        datasets = [self.test_sample[0][:sample_size], SDAE_denoised_noiseless_test, 
                    CDAE_denoised_noiseless_test, self.noisy_test_sample[0][:sample_size], 
                    SDAE_denoised_noisy_test, CDAE_denoised_noisy_test]
        names = ["Noiseless","Noiseless -> SDAE", "Noiseless -> CDAE",
                 "Noisy", "Noisy -> SDAE", "Noisy -> CDAE"]
        
        for i in range(sample_size):
            print("==================================="+str(i)+"=================================")
            for dataset, name in zip(datasets, names):
                print(name)
                plt.title("Label: " + self.label_names[self.dataset][int(labels[i])])
                plt.axis("off")
                plt.imshow(dataset[i], cmap = None if self.dataset == "CIFAR10" else "gray")
                plt.show()

def colab_env_check():
    '''
    Auxiliary function for running on colab. Prints any hardware accelerator settings
    and RAM size.
    '''

    from psutil import virtual_memory

    if tf.test.gpu_device_name(): 
        GPU_stats = tf.python.client.device_lib.list_local_devices()[-1].physical_device_desc
        print("You are running TF with the following GPU:\n", GPU_stats)
    else:
        try:
            tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  
            print('You are running on TPU ', tpu.cluster_spec().as_dict()['worker'])
            tf.config.experimental_connect_to_cluster(tpu)
            tf.tpu.experimental.initialize_tpu_system(tpu)
            tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)
        except:
            print("You are running TF without GPU or TPU...")
    ram_gb = virtual_memory().total / 1e9
    print('Your runtime has {:.1f} gigabytes of available RAM\n'.format(ram_gb))


####################################################################################################
# Running Code                                                                                     #
####################################################################################################

# Perform colab env check if needed
# colab_env_check()

# Current model only works with 3 datasets: "MNIST", Fashion MNIST("FMNIST") and CIFAR-10("CIFAR10")
for dataset in ["MNIST", "FMNIST", "CIFAR10"] :
    print("=============================DATASET:"+dataset+"================================")
    # Build ImageClassifier
    classifier = ImageClassifier()     

    # Load dataset. Use load_dataset(dataset, True) to show summary and example images
    classifier.load_dataset(dataset, False)

    # Add noise to the dataset. add_noise(True) to show examples of noisy images
    classifier.add_noise(False)

    # Preprocess data into noisy and noiseless trainign sets, as well as appropriate shape for 
    # each model
    classifier.preprocess_data()

    # TSNE visualization for clustering
    classifier.tsne()

    # Train denoisers, and show some samples for how a noisy image is denoised
    classifier.show_denoised_images()

    # Test 7 different optimizers (everything else are initial hyperparameters)
    classifier.train_models("Optimizers")

    # Test 7 different learning rates, using best optimizer found in the Optimizers test
    classifier.train_models("Learning Rate")

    # Test 7 different batch sizes, using best optimizers and learning rates found in previous 2 tests
    classifier.train_models("Batch Size")

    # Final test using best hyperparameters found in previous 3 tests
    classifier.train_models("Comparison")



