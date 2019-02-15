#!/usr/bin/env python


class BackgroundMatting(object):
    def get_mask(self, img):
        """
        Take an image of [H, W, 3]. Returns a mask of [H, W]
        """
        raise NotImplementedError()


class BackgroundMattingWithColor(BackgroundMatting):
    def __init__(self, color):
        """
        Args:
            color: a (r, g, b) tuple
        """
        self._color = color

    def get_mask(self, img):
        return img == self._color
