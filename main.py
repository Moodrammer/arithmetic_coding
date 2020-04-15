# Assignment 2: arithmetic coding
# Mahmoud Amr Mohamed Refaat
# sec : 2
# Bn : 25

import cv2
import os
import numpy as np
from numpy import save
from numpy import load

# -----------------------------------------------------------------------------------
# Enter a file name that is present in the same directory as main.py
# file = "blox.jpg"
file = input("Enter a file name that is present in the same directory as main.py (filename.ext) : ")
file_name, file_ext = os.path.splitext(file)
# -----------------------------------------------------------------------------------
# General parameters for encoding
# at blockSize = 16 underflow happens in most images except images having a low num of greyscale levels
# at blockSize = 8 underflow happens , but has a less effect
# at blockSize = 4 Images are retrieved successfully , but with larger size (x2) due to using float64 to store tags to
# prevent underflow
blockSize = int(input("Enter the block Size: "))
# precision of the encoded tags
encoded_prec = "float" + input("Enter the number of bits for float {16, 32, 64}: ")

print("/////////////////////////////////////////////////////////////////////")
print("Output Image will be saved as  Dec_filename_blocksize_Precision.extension")
print("/////////////////////////////////////////////////////////////////////")

# Read the image
img = np.array(cv2.imread(file, 0))

# Image dimensions
numRows = img.shape[0]
numCols = img.shape[1]
numZerosPadded = 0

# Flatten the image into a one dimensional array
img1D = img.reshape((1, img.size))

# Check if the number of pixels is divisible by the block size
if img1D.size % blockSize != 0:
    # Add code to handle image sizes that are non divisible by the blockSize
    print("It is not divisible by the blockSize")
    # Get the number of extra bits needed
    numZerosPadded = blockSize - img1D.size % blockSize
    # Create a temporary array to hold the new size
    temp = np.zeros((1, (img1D.size + numZerosPadded)), dtype=np.uint8)
    temp[0, 0:img1D.size] = img1D
    img1D = temp

# Calculate the frequencies
probList = np.zeros(256)

# Calculate the number of times each grey scale level appeared
for pixelIndex in range(0, img1D.size):
    probList[img1D[0, pixelIndex]] += 1

# Calculate the probabilities
probList /= img1D.size

# Encode
numBlocks = int(img1D.size/blockSize)

# np file to save the encoded image
encoded = np.zeros((1, numBlocks), dtype=encoded_prec)


def encode(left, right, currblock, blockindex, blocknumber):
    # Terminating condition
    if blockindex == blockSize:
        # Store the beginning of the final range
        encoded[0, blocknumber] = left
        return
    # Get the grey scale level to be decoded 0 - 255
    gs = currblock[blockindex]
    # get the needed rangebegin and rangeend
    probUpToBeg = probList[0:gs]
    Sumtobeg = probUpToBeg.sum()
    rangebegin = left + (right - left) * Sumtobeg
    rangeend = left + (right - left) * (Sumtobeg + probList[gs])

    encode(rangebegin, rangeend, currblock, blockindex + 1, blocknumber)


print("Encoding ...")
# Divide the image into blocks of the determined blockSize then encode each block and store the tag representing it
# in the encoded array
for blocknm in range(numBlocks):
    currBlock = img1D[0, blocknm * blockSize:(blocknm * blockSize + blockSize)]
    encode(0., 1., currBlock, 0, blocknm)

print("Encoded array of blocks")
print(encoded)
# Save the encoded list and the probability list to disk
save("encoded.npy", encoded)
save("problist.npy", probList)
##############################################################################################################
# Decoding
# Given :
# - The encoded image tags
# - The probabilty list of the original image grey scale levels
# - The original image dimensions (rows & Cols) [I get them from above but they should be sent to the decoder when it is
# isolated program
# - The blockSize used in encoding [Should also be sent to the decoder if it was an isolated program]
# - The number of zeroes padded at the end of the encoded image to handle being non divisible by the blockSize

# Load the required input for decoding from list
encodedImg = load('encoded.npy')
decprobList = load('problist.npy')
# Initialize array to hold the image as a 1D after decoding
decodedImg1D = np.empty((1, ((numRows * numCols) + numZerosPadded)), dtype=np.uint8)
# Initialize array to hold the block that is being decoded
decodedBlock = np.empty((1, blockSize), dtype=np.uint8)

# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


def decode(left, right, curr_code, block_index):
    # Terminating condition
    if block_index == blockSize:
        return
    # A variable to calculate the sum of all probabilities from the beginning of the probability list
    # to the beginning of the current range to be tested
    sum_to_beg = 0
    # Loop on all possible grey scale levels available from 0 to 255 to test whether the current tag is
    # within their range
    for gs in range(256):
        # calculate the beginning and end of all possible ranges
        range_begin = left + (right - left) * sum_to_beg
        sum_to_beg = sum_to_beg + decprobList[gs]
        range_end = left + (right - left) * sum_to_beg
        # In case of gs levels with probabilities equal to zero region_begin and region_end will be equal
        if range_begin < range_end:
            # If the current_code tag is within this range , store the decoded gs level and exit the loop
            if range_begin <= curr_code < range_end:
                decodedBlock[0, block_index] = gs
                break
    # Move to decoding the next element in the current block being decoded
    decode(range_begin, range_end, curr_code, block_index + 1)
# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


print("Decoding ...")
# Loop on all code tags each representing a 16 pixel block
# The beginning index of the current decoded block
for tagnum in range(encodedImg.size):
    # Get the current code word
    code = encodedImg[0, tagnum]
    decode(0., 1., code, 0)
    # Store the decoded block in its place within the decoded 1D image array
    decodedImg1D[0, tagnum * blockSize: tagnum * blockSize + blockSize] = decodedBlock[0]

# Remove the padded zeroes if any from encoding, if not its original value is zero
origSize = decodedImg1D.size - numZerosPadded
decImg1D_orig = decodedImg1D[0, 0:origSize]

# Restore the original shape of the image
decodedImg = decImg1D_orig.reshape((numRows, numCols))
print("File written to current directory")
# Save the decoded image to the current directory
cv2.imwrite('Dec_' + file_name + '_' + str(blockSize) + '_' + str(encoded_prec) + file_ext, decodedImg)
