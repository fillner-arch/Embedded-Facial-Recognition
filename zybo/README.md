 lbph.c

 Simple pure-C LBPH implementation (no OpenCV).

 Build:
   gcc -O2 -o lbph lbph.c -lm

 Usage:
   1) Prepare a training list file (train_list.txt) with lines:
      - label path_to_pgm_image
      -  0 faces/person0_1.pgm
      - 0 faces/person0_2.pgm
      - 1 faces/person1_1.pgm

   2) Run:
      ./lbph train_list.txt test_image.pgm

   It will load training images, compute LBPH features, compute feature for
   test image, and print best-matching label and chi-square distance.

 Notes:
  - Input images should be the same size (width x height). If not, resize offline.
  - Uses LBP with P=8 neighbors, R=1 and 256 histogram bins per cell.
  - Grid size defaults to 8x8 (change GRID_X/GIRD_Y constants or pass as args).
  - You can convert an image to pgm format using ImageMagick with the following command:
    - convert face.jpg -colorspace Gray -resize 128x128\! face.pgm