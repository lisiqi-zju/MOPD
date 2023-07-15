import glob
import os
import pdb
import argparse

def get_parser():
    parser = argparse.ArgumentParser(description="convert_coco")
    parser.add_argument(
        "--input_dir",
        default=f"../mask2d/output/opdmulti_V3_output_split/all/",
        metavar="DIR",
        help="directory of input data",
    )

    parser.add_argument(
        "--output_dir",
        default=f"../mask2d/output/MotionDataset_V3/",
        metavar="DIR",
        help="directory of output data",
    )

    return parser


def existDir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

if __name__ == "__main__":
    args = get_parser().parse_args()
    
    PROCESSPATH = args.input_dir
    DATASETPATH = args.output_dir

    # Create the dirs
    dir_names = ['train/', 'valid/', 'test/', 'annotations/', 'depth/']
    for dir_name in dir_names:
        existDir(DATASETPATH + dir_name)

    # Move the origin images and depth images
    origin_dir = ['train/', 'valid/', 'test/']
    for dir_name in origin_dir:
        print(f'Copying the {dir_name} images')

        # Move the origin images
        input_path = f'{PROCESSPATH}{dir_name}rgb/'
        # pdb.set_trace()
        output_path = f'{DATASETPATH}{dir_name}'
        # Loop the images
        file_paths = glob.glob(f'{input_path}*')
        for file_path in file_paths:
            file_name = file_path.split('/')[-1]
            new_file_path = f'{output_path}{file_name}'
            # pdb.set_trace()
            os.system(f'cp {file_path} {new_file_path}')

        # Move the depth images
        input_path = f'{PROCESSPATH}{dir_name}depth/'
        output_path = f'{DATASETPATH}depth/'
        # Loop the images
        file_paths = glob.glob(f'{input_path}*')
        for file_path in file_paths:
            file_name = file_path.split('/')[-1]
            new_file_path = f'{output_path}{file_name}'
            os.system(f'cp {file_path} {new_file_path}')

    # Move the annotations
    file_paths = glob.glob(f'{PROCESSPATH}coco_annotation/*')
    for file_path in file_paths:
        print('Copying the coco annotations')
        file_name = file_path.split('/')[-1]
        new_file_path = f'{DATASETPATH}annotations/{file_name}'
        os.system(f'cp {file_path} {new_file_path}')

