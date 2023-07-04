import tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from matplotlib import pyplot
from matplotlib.image import imread

# Rotação aleatória usando ImageDataGenerator 
training_data_generator = ImageDataGenerator(rotation_range=90, fill_mode='nearest')

# Diretório de imagens
training_image_directory = "/content/PRO_1-1_C125_HurricaneDamageDataset/train"

# Gere arquivos de imagem rotacionados aleatoriamente
training_augmented_images = training_data_generator.flow_from_directory(training_image_directory,target_size=(180,180))

# Aumento aleatório de dados (Redimensionamento, Deslocamento, Rotação) usando ImageDataGenerator 
training_data_generator = ImageDataGenerator(zoom_range=0.3, 
                                             horizontal_flip=True, 
                                             fill_mode='nearest')

# Diretório de Imagens
training_image_directory = "/content/PRO_1-1_C125_HurricaneDamageDataset/train"

# Gere arquivos de imagem aumentada
training_augmented_images = training_data_generator.flow_from_directory(
                                                       training_image_directory,
                                                       target_size=(180,180))

