from PIL import Image
import numpy as np
import math

###============================================================================###
#                   Edge detection using numpy arrays                            #
###============================================================================###
def n_edge_detection(kernel, kernel_secondary=None, detector=None):
    # Edge detection
    image = Image.open(r"sample.png")
    #image = Image.open(r"test.png")
    image = image.convert("L")
    arr = np.asarray(image)
    if detector == 'Prewitt':
        convolved = prewitt_convolve(kernel, arr, kernel_secondary)
    else:
        convolved = n_convolve_2d(kernel, arr)
    #convolved = n_convolve_2d_2(kernel, arr)
    im = Image.fromarray(convolved) #Bytes need to be scaled to between 0-255
    im.show()

def n_convolve_2d(kernel, static):
    "https://en.wikipedia.org/wiki/File:2D_Convolution_Animation.gif#/media/File:2D_Convolution_Animation.gif"
    static_shape = static.shape #(height, width)
    kernel_shape = kernel.shape #(height, width)
    kernel_width = kernel_shape[1]
    kernel_height = kernel_shape[0]
    padded_width = (static_shape[1] + (2 * math.floor((kernel_shape[1] / 2))))
    padded_height = (static_shape[0] + (2 * math.floor((kernel_shape[0] / 2))))
    padded_tuple = (padded_height, padded_width)
    static = n_pad_2d(static, padded_tuple, kernel_shape, static_shape)
    kernel = n_reverse_2d(kernel)
    row = np.array([])
    #have to start ranges at correct point
    #for 3x3 kernel this is [1,1]
    origin_width = math.floor(kernel_width/2)
    origin_height = math.floor(kernel_height/2)
    #iterate through rows then columns
    for j in range(origin_height, padded_height-origin_height):
        for i in range(origin_width, padded_width-origin_width):
            sum = 0
            for jj in range(kernel_height):
                for ii in range(kernel_width):
                    kY = ii
                    kX = jj
                    sX = j + jj - 1
                    sY = i + ii - 1
                    #print('kX: ', kX,'kY: ', kY,'sX: ', sX,'sY: ', sY)
                    #print('kernel: ', kernel[kX][kY])
                    #print('static: ', static[sX][sY])
                    sum += kernel[kX][kY] * static[sX, sY]
            row = np.append(row, sum)
    new = np.reshape(row, static_shape)
    print(new)
    return new

#@jit
def prewitt_convolve(kernel1, static, kernel2=None):
    #Start at 0,0 on both kernel and static
    #End at kernel_width-1 and kernel_height-1
    #append to a new array and reshape
    static_shape = static.shape  # (height, width)
    kernel_shape = kernel1.shape  # (height, width)
    new = np.array([])
    scale = 765 * 2#For a single Prewitt filter the maximum value possible is 765
    final_shape = (static_shape[0]+(1-kernel_shape[0]), static_shape[1]+(1-kernel_shape[1]))
    for sY in range(0, final_shape[0]): #Rows
        for sX in range(0, final_shape[1]): #Columns
            print('sY: ', sY, 'sX: ', sX)
            sum = 0
            for kY in range(kernel_shape[0]): #Rows
                for kX in range(kernel_shape[1]): #Columns:
                    #sum += (kernel1[kY][kX] * static[sY + kY, sX + kX])
                    sum += static[sY + kY, sX + kX] * (kernel1[kY][kX] + kernel2[kY][kX])
            scaled = sum/scale * 255
            new = np.append(new, abs(scaled))
    new = np.reshape(new, final_shape)
    print(new)
    return new

def n_reverse_2d(static):
    new = []
    for i in range(len(static)):
        row = static[i][::-1]
        new.insert(0, row)
    return new


def n_pad_2d(static, padded_tuple, kernel_tuple, static_tuple, filler=0):
    #Determine dimensions of kernel and static
    kernel_width = kernel_tuple[1]
    kernel_height = kernel_tuple[0]
    static_width = static_tuple[1]
    static_height = static_tuple[0]
    print ('kW: ', kernel_width, ' kH: ', kernel_height, ' sW: ', static_width, ' sH: ', static_height)
    #[rows][columns]
    oneD = np.array([])
    #vertical padding is kernel_height-1
    for j in range(math.floor(kernel_height/2)):
        for i in range(padded_tuple[1]):
            oneD = np.append(oneD, filler)
    for j in range(static_height):
        #for i in range(0, int((padded_tuple[1]-static_width)/2)):
        for i in range(0, math.floor(kernel_width/2)):
            oneD = np.append(oneD, filler)
        for i in range(0, static_width):
            print('j: ', j, ' i: ', i)
            oneD = np.append(oneD, static[j][i])
        for i in range(0, math.floor(kernel_width/2)):
        #for i in range(0, int((padded_tuple[1]-static_width)/2)):
            oneD = np.append(oneD, filler)
    for j in range(math.floor(kernel_height/2)):
        for i in range(padded_tuple[1]):
            oneD = np.append(oneD, filler)
    new = np.reshape(oneD, padded_tuple)
    print(new)
    return new



###============================================================================###
###                             Threading                                      ###
###============================================================================###
def main():

    #flatten()
    #kernel = [1, 2, 1]
    #static = [1, 2, 3, 4, 5]
    #kernel = [2, 0, -1, 2]
    #static = [-1, 0, 1]
    """
    kernel = [
        [1, 1, 1],
        [1, 5, 1],
        [1, 1, 1]
        ]
    """
    static = np.asarray( [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ] )

    kernel =np.asarray( [
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
    ] )
    laplacian_kernel =np.asarray( [
        [-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1]
    ] )
    Prewitt_horz = np.asarray([
        [1, 1, 1],
        [0, 0, 0],
        [-1, -1, -1]
    ])
    Prewitt_vert = np.asarray([
        [1, 0, -1],
        [1, 0, -1],
        [1, 0, -1]
    ])
    #conv2d.basic_convolve(kernel, static)
    #conv2d.convolve_2d(kernel, static)
    #edge_detection(laplacian_kernel)
    #n_edge_detection(Prewitt_horz, Prewitt_vert, 'Prewitt')
    #n_edge_detection(Prewitt_vert, Prewitt_horz, 'Prewitt')
    n_edge_detection(laplacian_kernel)
    #n_convolve_2d(kernel, static)

if __name__ == main():
    main()