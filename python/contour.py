from matplotlib.pyplot import contour
import numpy as np
import cv2 as cv


class ContourUtil(object):
    """_summary_

    ## Hierarch format
    [Next, previous, first_child, parent]

    Condition List
    self.hierarchy[:, 2] == -1 # the corresponding contour has no children
    self.hierarchy[:, 2] != -1 # the corresponding contour has at least 1 child


    Parameters
    ----------
    object : _type_
        _description_
    """

    def __init__(self, contours, hierarchy):
        self.contours = np.array(contours, dtype=object)
        self.hierarchy = np.squeeze(hierarchy)
        self.contours_ids = np.arange(len(contours))

    def _get_contour_with_condition(self, condition, contours):
        result_ids = np.argwhere(condition).flatten()
        result_contours = contours[condition]
        return result_contours, result_ids

    def _get_relevant(self, contour_ids):
        if contour_ids is None:
            return self.contours, self.contours_ids
        else:
            return self.contours[contour_ids], contour_ids

    def find_parents(self, contour_ids=None):
        contours, contour_ids = self._get_relevant(contour_ids)
        child_condition = self.hierarchy[contour_ids, 2] != -1
        return self._get_contour_with_condition(child_condition, contours)

    def are_parents(self, contour_ids=None):
        if contour_ids is None:
            contour_ids = np.arange(len(self.contours))

        return self.hierarchy[contour_ids, 2]

    def find_all_children(self, contour_idx):
        children_ids = self.hierarchy[self.hierarchy[:, -1]
                                      == contour_idx][:, 0]
        return self.contours[children_ids], children_ids

    def find_contour_with_no_children(self, contour_ids=None):
        contours, contour_ids = self._get_relevant(contour_ids)
        no_child_condition = self.hierarchy[contour_ids, 2] == -1
        return self._get_contour_with_condition(no_child_condition, contours)

    def compute_contour_stats(self, contour_ids=None):
        contours, contour_ids = self._get_relevant(contour_ids)
        areas = np.array([cv.contourArea(contour) for contour in contours])
        return areas, np.mean(areas), np.std(areas)

    @staticmethod
    def compute_stats(contour):
        pass

    @staticmethod
    def is_contour_quadrilateral(self, contour):
        return len(contour) == 4
