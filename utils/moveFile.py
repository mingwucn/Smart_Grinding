import os
import shutil
import re

def move_images(source_dir, dest_dir):
  """
  Moves images with the pattern 'defects_labels_cube{i}_layer{numbers}.png' 
  from the source directory to the corresponding 'cube{i}' subdirectory in the destination directory.

  Args:
    source_dir: The source directory containing the images.
    dest_dir: The destination directory.
  """
  for filename in os.listdir(source_dir):
    if filename.startswith('defects_labels_cube') and filename.endswith('.png'):
      cube_i = filename.split('_')[2].split('cube')[1]
      cube_dir = os.path.join(dest_dir, f"cube{cube_i}")
      os.makedirs(cube_dir, exist_ok=True)  # Create cube directory if it doesn't exist
      source_path = os.path.join(source_dir, filename)
      dest_path = os.path.join(cube_dir, filename.replace(f'_cube{cube_i}', ''))
      shutil.move(source_path, dest_path)

def move_npy_files(args):
  """
  Moves .npy files from the original structure to a new structure.

  Args:
    args: An object containing the necessary arguments, 
          specifically the 'cube_i' attribute.
  """
  source_dir = f"../lfs/point_wise_labels/cube{args.cube_i}"
  for root, dirs, files in os.walk(source_dir):
    for file in files:
      if file.endswith(".npy"):
        # Extract roi_radius from the filename
        match = re.search(r"layer\d+_roi_radius(\d+).npy", file)
        if match:
          roi_radius = match.group(1)
          dest_dir = f"../lfs/point_wise_labels/roi_radius{roi_radius}/cube{args.cube_i}"
          os.makedirs(dest_dir, exist_ok=True)  # Create destination directory if it doesn't exist
          source_path = os.path.join(root, file)
          dest_path = os.path.join(dest_dir, file.replace(f"_roi_radius{roi_radius}", ""))
          shutil.move(source_path, dest_path)
          print(f"Moved {source_path} to {dest_path}")

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description='Move file')
  parser.add_argument('--cube_i', type=int, default=2, help=f'Cube i, start from 0 (default: 2)')
  args = parser.parse_args()
  move_npy_files(args)


  # source_dir = "../outputs/img"
  # dest_dir = "../outputs/img"
  # move_images(source_dir, dest_dir)