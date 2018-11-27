import os
import warnings
from random import shuffle
from torch.autograd import Variable
from options.test_options import TestOptions
from models.models import create_model
import util.util as util
import torch
from PIL import Image
from skimage import io
import torchvision.transforms as transforms

opt = TestOptions().parse(save=False)
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1

ELEVATION_DIR = 'datasets/seamless_grid/test/elevation/'
LAND_MASK_DIR = 'datasets/seamless_grid/test/land_mask/'
LATITUDE_DIR = 'datasets/seamless_grid/test/latitude/'
GENERATED_IMAGE_DIR = 'datasets/seamless_grid/results/planet_mars_gen_2/generated_image/'

DATA_SIZE = [43200, 86400] # [y, x]
SLICE_SIZE = [1024, 1024]
STRIDE = [830, 830]

if opt.fineSize == 512:
    use_half_res = True
else:
    use_half_res = False

num_rows = DATA_SIZE[0] // STRIDE[0]
num_cols = DATA_SIZE[1] // STRIDE[1]
num_slices = num_rows * num_cols

slices_to_generate = []
generated_slices = set()

for row in range(num_rows):
    for col in range(num_cols):
        slices_to_generate.append(((row, col)))

shuffle(slices_to_generate)

model = create_model(opt)

for i, (row, col) in enumerate(slices_to_generate):
    fname = 'row_' + str(row) + '_col_' + str(col) + '.png'

    # Load Mars data:
    elevation = Image.open(ELEVATION_DIR + fname)
    elevation = elevation.resize((opt.fineSize, opt.fineSize), Image.LANCZOS)
    elevation = transforms.ToTensor()(elevation)
    elevation = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(elevation)

    land_mask = Image.open(LAND_MASK_DIR + fname)
    land_mask = land_mask.resize((opt.fineSize, opt.fineSize), Image.LANCZOS)
    land_mask = transforms.ToTensor()(land_mask)
    land_mask = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(land_mask)

    latitude = Image.open(LATITUDE_DIR + fname)
    latitude = latitude.resize((opt.fineSize, opt.fineSize), Image.LANCZOS)
    latitude = transforms.ToTensor()(latitude).type(torch.FloatTensor) / 64800
    latitude = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(latitude)

    # Prepare existing images:
    generated_image = torch.zeros((3, opt.fineSize, opt.fineSize), dtype=torch.float)
    generated_image_mask = torch.zeros((1, opt.fineSize, opt.fineSize), dtype=torch.float)

    # Calculate offset x and y:
    if row == 0:
        overlap_up = 0
        overlap_down = SLICE_SIZE[0] - STRIDE[0]
    elif row != num_rows - 1:
        overlap_up = SLICE_SIZE[0] - STRIDE[0]
        overlap_down = overlap_up
    else:
        offset_y = DATA_SIZE[0] - SLICE_SIZE[0]
        overlap_up = STRIDE[0] * (num_rows - 1) - offset_y
        overlap_down = 0
    
    if col == 0:
        normal_overlap_x = SLICE_SIZE[1] - STRIDE[1]
        half_remains = (DATA_SIZE[1] - STRIDE[1] * (num_cols)) // 2
        overlap_left = normal_overlap_x - half_remains
        overlap_right = SLICE_SIZE[1] - STRIDE[1]
    elif col != num_cols - 1:
        overlap_left = SLICE_SIZE[1] - STRIDE[1]
        overlap_right = overlap_left
    else:
        normal_overlap_x = SLICE_SIZE[1] - STRIDE[1]
        half_remains = (DATA_SIZE[1] - STRIDE[1] * (num_cols)) // 2
        offset_x = DATA_SIZE[1] - (STRIDE[1] + half_remains)
        overlap_left = STRIDE[1] * (num_cols - 1) - offset_x
        overlap_right = normal_overlap_x - half_remains
    
    if use_half_res:
        overlap_up //= 2
        overlap_down //= 2
        overlap_left //= 2
        overlap_right //= 2

    # Load existing images if exist:
    using_generated_images = False

    # Upper-left:
    if row > 0 and (row-1, col-1 if col-1>=0 else num_cols-1) in generated_slices:
        image = Image.open(GENERATED_IMAGE_DIR + 'row_' + str(row-1) + '_col_' + str(col-1 if col-1>=0 else num_cols-1) + '.png')
        image = transforms.ToTensor()(image)

        generated_image[:, :overlap_up, :overlap_left] = image[:, -overlap_up:, -overlap_left:]
        generated_image_mask[0, :overlap_up, :overlap_left] = 1

        using_generated_images = True

    # Upper-right:
    if row > 0 and (row-1, col+1 if col+1<num_cols else 0) in generated_slices:
        image = Image.open(GENERATED_IMAGE_DIR + 'row_' + str(row-1) + '_col_' + str(col+1 if col+1<num_cols else 0) + '.png')
        image = transforms.ToTensor()(image)

        generated_image[:, :overlap_up, -overlap_right:] = image[:, -overlap_up:, :overlap_right]
        generated_image_mask[0, :overlap_up, -overlap_right:] = 1

        using_generated_images = True

    # Lower-left:
    if row < num_rows-1 and (row+1, col-1 if col-1>=0 else num_cols-1) in generated_slices:
        image = Image.open(GENERATED_IMAGE_DIR + 'row_' + str(row+1) + '_col_' + str(col-1 if col-1>=0 else num_cols-1) + '.png')
        image = transforms.ToTensor()(image)

        generated_image[:, -overlap_down:, :overlap_left] = image[:, :overlap_down, -overlap_left:]
        generated_image_mask[0, -overlap_down:, :overlap_left] = 1

        using_generated_images = True
    
    # Lower-right:
    if row < num_rows-1 and (row+1, col+1 if col+1<num_cols else 0) in generated_slices:
        image = Image.open(GENERATED_IMAGE_DIR + 'row_' + str(row+1) + '_col_' + str(col+1 if col+1<num_cols else 0) + '.png')
        image = transforms.ToTensor()(image)

        generated_image[:, -overlap_down:, -overlap_right:] = image[:, :overlap_down, :overlap_right]
        generated_image_mask[0, -overlap_down:, -overlap_right:] = 1

        using_generated_images = True
    
    # Upper:
    if row > 0 and (row-1, col) in generated_slices:
        image = Image.open(GENERATED_IMAGE_DIR + 'row_' + str(row-1) + '_col_' + str(col) + '.png')
        image = transforms.ToTensor()(image)

        generated_image[:, :overlap_up, :] = image[:, -overlap_up:, :]
        generated_image_mask[0, :overlap_up, :] = 1

        using_generated_images = True

    # Lower:
    if row < num_rows-1 and (row+1, col) in generated_slices:
        image = Image.open(GENERATED_IMAGE_DIR + 'row_' + str(row+1) + '_col_' + str(col) + '.png')
        image = transforms.ToTensor()(image)

        generated_image[:, -overlap_down:, :] = image[:, :overlap_down, :]
        generated_image_mask[0, -overlap_down:, :] = 1

        using_generated_images = True

    # Left:
    if (row, col-1 if col-1>=0 else num_cols-1) in generated_slices:
        image = Image.open(GENERATED_IMAGE_DIR + 'row_' + str(row) + '_col_' + str(col-1 if col-1>=0 else num_cols-1) + '.png')
        image = transforms.ToTensor()(image)

        generated_image[:, :, :overlap_left] = image[:, :, -overlap_left:]
        generated_image_mask[0, :, :overlap_left] = 1

        using_generated_images = True

    # Right:
    if (row, col+1 if col+1<num_cols else 0) in generated_slices:
        image = Image.open(GENERATED_IMAGE_DIR + 'row_' + str(row) + '_col_' + str(col+1 if col+1<num_cols else 0) + '.png')
        image = transforms.ToTensor()(image)

        generated_image[:, :, -overlap_right:] = image[:, :, :overlap_right]
        generated_image_mask[0, :, -overlap_right:] = 1

        using_generated_images = True

    generated_image = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(generated_image)
    generated_image_mask = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(generated_image_mask)

    # Generate Mars image slice:
    if using_generated_images:
        print('Generating row {}, column {} ({}/{}), with existing images'.format(row, col, i, num_slices))
    else:
        print('Generating row {}, column {} ({}/{})'.format(row, col, i, num_slices))

    A_tensor = torch.cat((elevation, land_mask, latitude, generated_image, generated_image_mask), dim=0).unsqueeze(0)
    generated_tensor = model.inference(A_tensor)
    generated_image = util.tensor2im(generated_tensor.data[0])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        io.imsave(GENERATED_IMAGE_DIR + fname, generated_image)

    generated_slices.add((row, col))