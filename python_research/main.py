import os
import cv2
import numpy as np
from contour import ContourUtil
import logging
import json

with open("configs/config.json", 'r') as f:
    config = json.load(f)

approximation_epsilon = config["approximation_epsilon"]
min_contour_proportion = config["minimum_contour_area_proportion"]
rows, columns = config['frame_shape']
frame_area = rows*columns

log_format = "%(levelname)s [%(lineno)s] [%(message)s]"
logging.basicConfig(format=log_format, level=logging.DEBUG)
logger = logging.getLogger(__name__)

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


def find_potential_sudoku_grid(contour_utils, parent_ids):
    contours_of_interest = []
    for current_contour_id in parent_ids:
        # the contour which has at least 1 children
        parent_contour = contour_utils.contours[current_contour_id]

        # index of children of the parent contour
        _, children_ids = contour_utils.find_all_children(current_contour_id)
        neighbors = len(children_ids)

        # total children that are also parents (make parent_contour grand parent)
        _, child_parent_ids = contour_utils.find_parents(children_ids)
        total_child_parents = len(child_parent_ids)

        if neighbors > 1 and total_child_parents > 1:
            contours_of_interest.append(parent_contour)

    poly_contours = []
    # get approximate polygon of the contours
    approx_contours= []
    
    max_contour_area = float('-inf')
    #check if area of the approximate contour is at least 10% of frame
    # and select the contour with max area
    for contour in contours_of_interest:
        epsilon = approximation_epsilon * cv2.arcLength(contour, closed=False)
        con = cv2.approxPolyDP(contour, epsilon=epsilon, closed=True)
        approx_contours.append(con)
        if (contour_area := cv2.contourArea(con)) > (min_contour_proportion * frame_area) and \
                contour_area > max_contour_area and \
                len(con) == 4:
            poly_contours.append(con)
            max_contour_area = contour_area
    
    #ensure contour is almost a rectangle is 
    try:
        corners = find_corners(poly_contours[0])
        if not is_rect(*corners):
            logger.warning("Not a Rectangle")
            poly_contours = []
            corners = []
    except IndexError:
        corners = []

    # TODO: select contour with the least shift
    zero_img = np.zeros((rows, columns), dtype = np.uint8)
    cv2.drawContours(zero_img, approx_contours, -1, color = 255, thickness=1)
    cv2.imshow("approx", zero_img)
    return poly_contours, corners


def draw_lines(image, lines, color=(0, 0, 255)):
    if lines is None:
        return image
    lines = lines.reshape((-1, 4))
    for line in lines:
        pt1 = (line[0], line[1])
        pt2 = (line[2], line[3])

        cv2.line(image, pt1, pt2, color, 1, cv2.LINE_AA)

    return image


def warp_sudoku_grid(frame, gray_frame, points):
    pts2 = np.float32([[0, 0], [499, 0],
                       [0, 499], [499, 499]])
    matrix = cv2.getPerspectiveTransform(np.float32(points), pts2)
    result = cv2.warpPerspective(frame, matrix, (500, 500))
    return result


def find_grid(video_path):

    if os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
    else:
        logger.info(f"File {video_path} doesn't exist. Using webcam")
        cap = cv2.VideoCapture(0)

    gaussian_kernel_size = config['gaussian_kernel_size']
    gaussian_sigma = config['gaussian_sigma']
    canny_threshold_1 = config["canny_min_threshold"]
    canny_threshold_2 = config["canny_max_threshold"]
    canny_aperture_size = config["canny_aperture_size"]
    hough_config = config["hough"]
    hough_threshold = hough_config['threshold']
    hough_rho = hough_config["rho"]
    hough_theta = hough_config["theta"]
    hough_min_length = hough_config["min_line_length"]
    hough_max_gap = hough_config["max_line_gap"]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # blur_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # high sigma needed because only interested in edges
        blur_frame = cv2.GaussianBlur(
            gray_frame, gaussian_kernel_size, gaussian_sigma)

        blur_frame = cv2.Canny(blur_frame, canny_threshold_1, canny_threshold_2,
                               apertureSize=canny_aperture_size, L2gradient=True)

        # Sudoku grids are only composed of lines
        lines = cv2.HoughLinesP(blur_frame, hough_rho, hough_theta,
                                hough_threshold, None, hough_min_length, hough_max_gap)
        blur_frame = draw_lines(blur_frame, lines, 255)
        cv2.imshow("blur_frame", blur_frame)
        cv2.imshow("gray_frame", gray_frame)

        # cv2.imshow("frame", frame)
        
        contours, hierarchy = cv2.findContours(
            blur_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contour_utils = ContourUtil(contours, hierarchy)

        if not contours:
            continue

        hierarchy = np.squeeze(hierarchy)
        contours = np.array(contours, dtype=object)

        parent_contours, parent_ids = contour_utils.find_parents()
        grids, grid_corners = find_potential_sudoku_grid(
            contour_utils, parent_ids)

        contour_frame = np.zeros_like(frame)
        cv2.drawContours(contour_frame, contourIdx=-1,
                         contours=contours, color=(0, 255, 0))
        parent_contour_frame = np.zeros_like(frame)
        cv2.drawContours(parent_contour_frame, contourIdx=-1,
                         contours=parent_contours, color=(255, 0, 0))
        grid_frame = frame.copy()
        cv2.drawContours(grid_frame, contourIdx=-1,
                         contours=grids, color=(0, 255, 255))
        if grid_corners:
            top_left, top_right, bottom_right, bottom_left = grid_corners
            cv2.circle(grid_frame, center=top_left, radius=5,
                       color=config["colors"]["RED"], thickness=-1)
            cv2.circle(grid_frame, center=top_right, radius=5,
                       color=config["colors"]["GREEN"],  thickness=-1)
            cv2.circle(grid_frame, center=bottom_right, radius=5,
                       color=config["colors"]["BLUE"],  thickness=-1)
            cv2.circle(grid_frame, center=bottom_left, radius=5,
                       color=config["colors"]["WHITE"],  thickness=-1)

            warped_frame = warp_sudoku_grid(frame, gray_frame, [top_left, top_right, bottom_left, bottom_right])
            cv2.imshow("warped_frame", warped_frame)
            key_2 = cv2.waitKey(0)

            if key_2 == ord('s') & 0xff or key_2 == ord("s") & 0xff:
                cv2.imwrite("python/detected_grid.png", warped_frame)
        # cv2.imshow("frame", frame)
        # cv2.imshow("contour_frame", contour_frame)
        cv2.imshow("parent_contour_frame", parent_contour_frame)
        cv2.imshow("grid_frame", grid_frame)


        key = cv2.waitKey(1)
        if key == ord('q') & 0xff or key == ord("Q") & 0xff:
            break

        if key == ord('p') & 0xff or key == ord('P') & 0xff:
            cv2.waitKey(0)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    find_grid(config["video_path"])

