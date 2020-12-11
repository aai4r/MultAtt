from PIL import Image
from autocrop import Cropper
import glob
import os

cropper = Cropper()

count = 0
for file in glob.iglob(r'PATH/TO/DATASET/CAER-S/train/*/*.png', recursive=True):
    count += 1
    print('{0:5d}/{1:5d}'.format(count, len(glob.glob(r'PATH/TO/DATASET/CAER-S/train/*/*.png'))))

    new_file = file.split('/')
    new_file[-3] += '_preprocessed'
    if not os.path.exists('/'.join(new_file[:-2])):
        os.makedirs('/'.join(new_file[:-2]))

    try:
        # Get a Numpy array of the cropped image
        cropped_array, masked_array = cropper.crop(file)

        # Save the cropped image with PIL
        cropped_image = Image.fromarray(cropped_array)
        masked_image = Image.fromarray(masked_array)

        if not os.path.exists('/'.join(new_file[:-1])):
            os.makedirs('/'.join(new_file[:-1]))

        cropped_image.save(os.path.join('/'.join(new_file[:-1]), file[-8:-4] + '_cropped.png'))
        masked_image.save(os.path.join('/'.join(new_file[:-1]), file[-8:-4] + '_masked.png'))
    except:
        if not os.path.exists('/'.join(new_file[:-1]) + '_failed'):
            os.makedirs('/'.join(new_file[:-1]) + '_failed')

        image = cropper.failed_file(file)
        image = Image.fromarray(image)
        image.save(os.path.join('/'.join(new_file[:-1]) + '_failed', file[-8:]))

count = 0
for file in glob.iglob(r'PATH/TO/DATASET/CAER-S/test/*/*.png', recursive=True):
    count += 1
    print('{0:5d}/{1:5d}'.format(count, len(glob.glob(r'PATH/TO/DATASET/CAER-S/test/*/*.png'))))

    new_file = file.split('/')
    new_file[-3] += '_preprocessed'
    if not os.path.exists('/'.join(new_file[:-2])):
        os.makedirs('/'.join(new_file[:-2]))

    try:
        # Get a Numpy array of the cropped image
        cropped_array, masked_array = cropper.crop(file)

        # Save the cropped image with PIL
        cropped_image = Image.fromarray(cropped_array)
        masked_image = Image.fromarray(masked_array)

        if not os.path.exists('/'.join(new_file[:-1])):
            os.makedirs('/'.join(new_file[:-1]))

        cropped_image.save(os.path.join('/'.join(new_file[:-1]), file[-8:-4] + '_cropped.png'))
        masked_image.save(os.path.join('/'.join(new_file[:-1]), file[-8:-4] + '_masked.png'))
    except:
        if not os.path.exists('/'.join(new_file[:-1]) + '_failed'):
            os.makedirs('/'.join(new_file[:-1]) + '_failed')

        image = cropper.failed_file(file)
        image = Image.fromarray(image)
        image.save(os.path.join('/'.join(new_file[:-1]) + '_failed', file[-8:]))
