#!/usr/bin/env python

import numpy as np
import cv2
import skvideo.io

class ImageSource(object):
    def get_image(self):
        """
        Returns:
            an RGB image of [h, w, 3] with a fixed shape.
        """
        pass

    def reset(self):
        """ Called when an episode ends. """
        pass


class FixedColorSource(ImageSource):
    def __init__(self, shape, color):
        """
        Args:
            shape: [h, w]
            color: (r, g, b)
        """
        self.arr = np.zeros((shape[0], shape[1], 3))
        self.arr[:, :] = color

    def get_image(self):
        return np.copy(self.arr)


class RandomColorSource(ImageSource):
    def __init__(self, shape):
        """
        Args:
            shape: [h, w]
        """
        self.shape = shape
        self.reset()

    def reset(self):
        self._color = np.random.randint(0, 256, size=(3,))

    def get_image(self):
        arr = np.zeros((self.shape[0], self.shape[1], 3))
        arr[:, :] = self._color
        return arr


class RandomImageSource(ImageSource):
    def __init__(self, shape, filelist):
        """
        Args:
            shape: [h, w]
            filelist: a list of image files
        """
        self.shape_wh = shape[::-1]
        self.filelist = filelist
        self.reset()

    def reset(self):
        fname = np.random.choice(self.filelist)
        im = cv2.imread(fname, cv2.IMREAD_COLOR)
        im = im[:, :, ::-1]
        im = cv2.resize(im, self.shape_wh)
        self._im = im

    def get_image(self):
        return self._im


class RandomVideoSource(ImageSource):
    def __init__(self, shape, filelist, fps=5):
        """
        Args:
            shape: [h, w]
            filelist: a list of video files
        """
        self.shape_wh = shape[::-1]
        self.filelist = filelist
        self._vc = None
        self.fps = fps
        self.reset()

    def reset(self):
        if self._vc is not None:
            self._vc.release()

        fname = np.random.choice(self.filelist)
        # self._vc = cv2.VideoCapture(fname)
        # self._vc.set(cv2.CAP_PROP_FPS, self.fps)
        self.frames = skvideo.io.vreader(fname)

    def get_image(self):
        try:
            im = next(self.frames)
        except StopIteration:
            self.reset()
            im = next(self.frames)
        im = im[:, :, ::-1]
        im = cv2.resize(im, self.shape_wh)
        return im
