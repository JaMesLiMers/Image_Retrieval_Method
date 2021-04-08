from typing import Any
import numpy as np
import itertools

class ColorHistogram():
    def __init__(self, n_bin=12, h_type='region', n_slice=3, depth=3, normalize=True) -> None:
        """
        Define the basic configuration:

        Args:
            n_bin: (default: 12)
                number of bins for each channel
            h_type: (default: 'region')
                "global" or "region":
                    'global' means count the histogram for whole image
                    'region' means count the histogram for regions in images, then concatanate all of them
            n_slice: (default: 3)
                Slice of image, Work when type equals to 'region', height & width will equally sliced into N slices
            depth: (default: 3)
                retrieved depth, set to None will count the ap for whole database
            normalize: (default: True)
                normalize output histogram
        
        """
        # configs for histogram
        self.n_bin   = n_bin            # histogram bins
        self.n_slice = n_slice          # slice image
        self.h_type  = h_type           # global or region
        self.depth   = depth            # retrieved depth, set to None will count the ap for whole database
        self.normalize = normalize      # normalize output histogram or not

    def histogram(self, input):
        """ Count img color histogram
  
        Args:
            input (numpy.ndarray)
                image in form of numpy.ndarray, Image shape should be [height, width, channel].
        Return
            h_type == 'global'
                a numpy array with size n_bin ** channel
            h_type == 'region'
                a numpy array with size n_slice * n_slice * (n_bin ** channel)
        """
        # examinate input type
        assert isinstance(input, np.ndarray), "input should be a ndarray"  
        img = input.copy()

        # image shape
        height, width, channel = img.shape
        # slice bins equally for each channel (颜色量化)
        bins = np.linspace(0, 256, self.n_bin+1, endpoint=True)

        
        if self.h_type == 'global':
            # Count hist derectly
            hist = self._count_hist(img, self.n_bin, bins, channel)
        elif self.h_type == 'region':
            # Count for every img bins:
            # Init Place holders
            hist = np.zeros((self.n_slice, self.n_slice, self.n_bin ** channel))
            # slice all block index
            h_silce = np.around(np.linspace(0, height, self.n_slice+1, endpoint=True)).astype(int)
            w_slice = np.around(np.linspace(0, width, self.n_slice+1, endpoint=True)).astype(int)
            for hs in range(len(h_silce)-1):
                for ws in range(len(w_slice)-1):
                    # slice img to regions by index
                    img_r = img[h_silce[hs]:h_silce[hs+1], w_slice[ws]:w_slice[ws+1]]
                    hist[hs][ws] = self._count_hist(img_r, self.n_bin, bins, channel)
        if self.normalize:
            hist /= np.sum(hist)

        return hist.flatten()

    def _count_hist(self, input, n_bin, bins, channel):
        """ Count hist for every bins

        Input example:
            [[256, 0,]
            [128, 0,]]
            
        Output example: (with n_bin = 4, bins = [0, 128, 256], channel = 1)
            [2. 2. 0. 0.]

        Args:
            input (numpy.ndarray)
                image in form of numpy.ndarray
            n_bin: (int)
                number of bins for each channel
            bins: (numpy.ndarray)
                image bins, array of color value(0-255)
            channel: (int)
                Image channel

        Return:
            A numpy array with size n_bin ** channel
            Stand for classification of every bins to color.
            
        """
        img = input.copy()
        bins_idx = {key: idx for idx, key in enumerate(itertools.product(np.arange(n_bin), repeat=channel))}  # permutation of bins
        hist = np.zeros(n_bin ** channel)
    
        # cluster every pixels
        for idx in range(len(bins)-1):
            img[(input >= bins[idx]) & (input < bins[idx+1])] = idx
        # add pixels into bins
        height, width, _ = img.shape
        for h in range(height):
            for w in range(width):
                b_idx = bins_idx[tuple(img[h,w])]
                hist[b_idx] += 1
        return hist


    def forward(self, input):
        return self.histogram(input)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        self.forward(self, args, kwds)


if __name__ == "__main__":
    a = ColorHistogram()

    test_img = np.array([[255, 0,],[128, 0,]])
    test_img = test_img[:, :, np.newaxis]

    print("Input: {}".format(list(test_img)))

    print("Output: {}".format(a._count_hist(test_img, 4, [0,128, 256], 1)))