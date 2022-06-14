import cv2
import numpy as np

def draw_lines(image, lines, color=(0, 0, 255)):
    if lines is None:
        return image
    lines = lines.reshape((-1, 4))
    for line in lines:
        pt1 = (line[0], line[1])
        pt2 = (line[2], line[3])

        cv2.line(image, pt1, pt2, color, 1, cv2.LINE_AA)

    return image

def find_corners(contour):
        assert len(contour) == 4, "The contour must only have 4 points to find corners"
        contour = contour.squeeze()
        x_argsort = np.argsort(contour, axis=0)[:, 0]

        if (l_1 := contour[x_argsort[0]])[1] < (l_2 := contour[x_argsort[1]])[1]:
            top_left, bottom_left = l_1, l_2
        else:
            top_left, bottom_left = l_2, l_1

        if (r_1 := contour[x_argsort[2]])[1] < (r_2 := contour[x_argsort[3]])[1]:
            top_right, bottom_right = r_1, r_2
        else:
            top_right, bottom_right = r_2, r_1

        return top_left, top_right, bottom_right, bottom_left


def similar_length(l_1, l_2, min_prop = 0.8, max_prop=1.2):
    big_length, small_length = (l_1, l_2) if l_1>l_2 else (l_2, l_1)
    return min_prop * big_length < small_length < max_prop * big_length
        
def is_rect(top_left, top_right, bottom_right, bottom_left):
    """Checks if the corners of the quadrilateral form an almost rectangle

    Parameters
    ----------
    top_left : _type_
        _description_
    top_right : _type_
        _description_
    bottom_right : _type_
        _description_
    bottom_left : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    return similar_length(top_right[0] - top_left[0], bottom_right[0]-bottom_left[0]) and \
        similar_length(bottom_right[1]-top_right[1], bottom_left[1] - top_left[1])
    