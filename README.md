# cnn_ni

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg 	

###Start code here
img = plt.imread('home.png')
data =img.reshape(1, 252, 362, 3)
###End code

print(type(img))
print("Image dimension ",img.shape)
print("Input data dimension ", data.shape)    


plt.imshow(data[0,:,:,:])
plt.grid(False)
plt.axis("off")


def zero_pad(data, pad):
    ###Start code here
    data_padded =  np.pad(data, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant', constant_values=(0, 0))
    return data_padded
    ###End code
	
	
print("dimension before padding: ", data.shape)
img_pad = zero_pad(data, 10)
print("dimension after padding: ", img_pad.shape)
print(img_pad[0,8:12,8:12,1])
plt.imshow(img_pad[0,:,:,:], cmap = "gray")
plt.grid(False)

output1 = np.mean(img_pad)


def conv_single_step(data_slice, W, b):
    ###Start code
    conv =  W * data_slice +b
    Z = np.sum(conv)
   # Z = Z + float(b)
    ###End code
    return Z
	
	
def conv_forward(data, W, b, hparams):
    ###Start code here
    # Retrieve dimensions from A_prev's shape (≈1 line)
    (m, n_H_prev, n_W_prev, n_C_prev) = data.shape

    # Retrieve dimensions from W's shape (≈1 line)
    (f, f, n_C_prev, n_C) = W.shape

    # Retrieve information from "hparameters" (≈2 lines)
    stride = hparams['stride']
    pad = hparams['pad']

    # Compute the dimensions of the CONV output volume using the formula given above. Hint: use int() to floor. (≈2 lines)
    n_H = int((n_H_prev - f + (2 * pad)) / stride + 1)
    n_W = int((n_W_prev - f + (2 * pad)) / stride + 1)

    # Initialize the output volume Z with zeros. (≈1 line)
    Z = np.zeros((m, n_H, n_W, n_C))

    # Create A_prev_pad by padding A_prev
    A_prev_pad = zero_pad(data, pad)

    for i in range(m):  # loop over the batch of training examples
        a_prev_pad = data[i]  # Select ith training example's padded activation
        for h in range(n_H):  # loop over vertical axis of the output volume
            for w in range(n_W):  # loop over horizontal axis of the output volume
                for c in range(n_C):  # loop over channels (= #filters) of the output volume

                    # Find the corners of the current "slice" (≈4 lines)
                    vert_start = stride * h
                    vert_end = stride * h + f
                    horiz_start = stride * w
                    horiz_end = stride * w + f

                    # Use the corners to define the (3D) slice of a_prev_pad (See Hint above the cell). (≈1 line)
                    a_slice_prev = A_prev_pad[i, vert_start:vert_end, horiz_start:horiz_end, :]

                    # Convolve the (3D) slice with the correct filter W and bias b, to get back one output neuron. (≈1 line)
                    Z[i, h, w, c] = conv_single_step(a_slice_prev, W[:, :, :, c], b[:, :, :, c])
    
   

    ###End code
    return Z ##(convolved output)
	
np.random.seed(1)
input_ = np.random.randn(10, 4, 4, 3)
W = np.random.randn(2, 2, 3, 8)
b = np.random.randn(1, 1, 1, 8)
hparameters = {"pad" : 1,
               "stride": 1}

output_ = conv_forward(input_, W, b, hparams)
print(np.mean(output_))


###Start code
hparams = {"pad" :1, "stride": 1}
b = np.zeros(4)
Z = conv_forward(edge_detect, b, hparams)

plt.clf()
plt.imshow(Z[0,:,:,0], cmap='gray',vmin=0, vmax=1)
plt.grid(False)
print("dimension of image before convolution: ", data.shape)
print("dimension of image after convolution: ", Z.shape)

output2 = np.mean(Z[0,100:200,200:300,0])
