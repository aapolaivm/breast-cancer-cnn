def resize_image(image, target_size):
    from keras.preprocessing.image import img_to_array, load_img
    image = load_img(image, target_size=target_size)
    return img_to_array(image)

def normalize_image(image):
    return image.astype('float32') / 255.0

def augment_data(image):
    from keras.preprocessing.image import ImageDataGenerator
    datagen = ImageDataGenerator(rotation_range=20,
                                 width_shift_range=0.2,
                                 height_shift_range=0.2,
                                 shear_range=0.2,
                                 zoom_range=0.2,
                                 horizontal_flip=True,
                                 fill_mode='nearest')
    return datagen.flow(image)