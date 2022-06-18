import numpy as np
import cv2
from contour import ContourUtil
import os
import utils
import logging
from python_research import Color
import math

logger = logging.getLogger(__name__)


class SudokuScanner(object):
    def __init__(self, config: dict):
        # self.config = config
        if os.path.exists(video_path := (config["video_path"])):
            self.cap = cv2.VideoCapture(video_path)
        else:
            self.cap = cv2.VideoCapture(0)

        hough_config = config["hough"]
        grid_config = config['grid_param']
        self.gaussian_kernel_size = config['gaussian_kernel_size']
        self.gaussian_sigma = config['gaussian_sigma']
        self.canny_threshold_1 = config["canny_min_threshold"]
        self.canny_threshold_2 = config["canny_max_threshold"]
        self.canny_aperture_size = config["canny_aperture_size"]
        self.hough_config = config["hough"]
        self.hough_threshold = hough_config['threshold']
        self.hough_rho = hough_config["rho"]
        self.hough_theta = hough_config["theta"]
        self.hough_min_length = hough_config["min_line_length"]
        self.hough_max_gap = hough_config["max_line_gap"]
        self.approximation_epsilon = config["approximation_epsilon"]
        self.min_contour_proportion = config["minimum_contour_area_proportion"]
        self.warp_img_height = config["warp"]['height']
        self.warp_img_width = config["warp"]['width']

        self.grid_gaussian_kernel_size = config['gaussian_kernel_size']
        self.grid_gaussian_sigma = grid_config['gaussian_sigma']
        self.grid_canny_threshold_1 = grid_config["canny_min_threshold"]
        self.grid_canny_threshold_2 = grid_config["canny_max_threshold"]
        self.grid_canny_aperture_size = grid_config["canny_aperture_size"]

    def distance(self, p, q):
        return math.sqrt((p[0]-q[0])**2 + (p[1]-q[1])**2)

    def compute_bounding_diagonals(self, contours):
        distances = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            distances.append(self.distance((x, y), (x+w, y+h)))
        return distances

    def preprocess(self, img):
        gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur_frame = cv2.GaussianBlur(
            gray_frame, self.gaussian_kernel_size, self.gaussian_sigma)

        return gray_frame, blur_frame

    def draw_bounding_rects(self, img, contours, color = Color.WHITE.value):
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(img, (x,y), (x+w, y+h), color = color, thickness=1)
        return img


    def find_grid_cells(self, img: np.ndarray):
        gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_area = gray_frame.shape[0] * gray_frame.shape[1]
        blur_frame = cv2.GaussianBlur(
            gray_frame, self.grid_gaussian_kernel_size, self.grid_gaussian_sigma)
        edge_image = cv2.Canny(blur_frame, self.grid_canny_threshold_1, self.grid_canny_threshold_2,
                               apertureSize=self.grid_canny_aperture_size, L2gradient=True)

        edge_image = cv2.morphologyEx(edge_image, cv2.MORPH_DILATE, np.ones(
            (3, 3), dtype=np.uint8,), iterations=1)

        contours, hierarchy = cv2.findContours(
            edge_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        contour_util = ContourUtil(contours, hierarchy)
        contours = np.array(contours, dtype=object)
        hierarchy = np.squeeze(hierarchy)

        child_contours, child_contour_ids = contour_util.find_contour_with_no_children()

        contour_areas, _, _ = contour_util.compute_contour_stats(
            child_contour_ids)

        zero_img = np.zeros_like(img)

        max_area_condition = contour_areas < (0.012 * img_area)

        sorted_areas = np.sort(
            contour_areas[max_area_condition])
        diff = np.diff(sorted_areas)
        area_partition_value = sorted_areas[np.argwhere(
            diff == diff.max())[0].item()]

        partition_condition = (contour_areas <= area_partition_value) & max_area_condition

        number_areas = contour_areas[partition_condition]
        number_area_mean = np.median(number_areas)
        number_area_std = np.std(number_areas)

        condition = (contour_areas <= (number_area_mean + (1 * number_area_std))
                     ) & (contour_areas >= (number_area_mean - (1 * number_area_std)))

        number_contours = child_contours[condition]


        number_contour_diagonals = np.array(self.compute_bounding_diagonals(number_contours))
        max_diagonal = 0.12 * self.distance((0, 0), (500, 500))
        
        diagonal_condition = number_contour_diagonals <= max_diagonal
        final_numbers = number_contours[diagonal_condition]
        
        #TODO: identify the coordinates where numbers are detected
        #detected contours - green
        cv2.drawContours(zero_img, contourIdx=-1,
                            contours=contours, color=Color.GREEN.value)
        #contours at bottom of hierarchy - blue 
        cv2.drawContours(zero_img, contourIdx=-1,
                            contours=child_contours, color= Color.BLUE.value)
        
        #contours predicted as numers - red
        cv2.drawContours(zero_img, contourIdx=-1,
                            contours=final_numbers, color=Color.RED.value)

        self.draw_bounding_rects(zero_img, final_numbers)

        return zero_img, final_numbers

    def find_straight_edges(self, img: np.ndarray) -> np.ndarray:
        blur_frame = cv2.Canny(img, self.canny_threshold_1, self.canny_threshold_2,
                               apertureSize=self.canny_aperture_size, L2gradient=True)

        # Sudoku grids are only composed of lines
        lines = cv2.HoughLinesP(blur_frame, self.hough_rho, self.hough_theta,
                                self.hough_threshold, None, self.hough_min_length, self.hough_max_gap)
        blur_frame = utils.draw_lines(blur_frame, lines, 255)
        return blur_frame

    def _find_sudoku_from_contours(self, contour_utils, img_shape):
        rows, columns = img_shape[:2]
        frame_area = rows * columns
        _, parent_ids = contour_utils.find_parents()
        contours_of_interest = []
        for current_contour_id in parent_ids:
            # the contour which has at least 1 children
            parent_contour = contour_utils.contours[current_contour_id]

            # index of children of the parent contour
            _, children_ids = contour_utils.find_all_children(
                current_contour_id)
            neighbors = len(children_ids)

            # total children that are also parents (make parent_contour grand parent)
            _, child_parent_ids = contour_utils.find_parents(children_ids)
            total_child_parents = len(child_parent_ids)

            if neighbors > 1 and total_child_parents > 1:
                contours_of_interest.append(parent_contour)

        poly_contours = []
        # get approximate polygon of the contours
        approx_contours = []

        max_contour_area = float('-inf')
        # check if area of the approximate contour is at least 10% of frame
        # and select the contour with max area
        for contour in contours_of_interest:
            epsilon = self.approximation_epsilon * \
                cv2.arcLength(contour, closed=False)
            con = cv2.approxPolyDP(contour, epsilon=epsilon, closed=True)
            approx_contours.append(con)
            if (contour_area := cv2.contourArea(con)) > (self.min_contour_proportion * frame_area) and \
                    contour_area > max_contour_area and \
                    len(con) == 4:
                poly_contours.append(con)
                max_contour_area = contour_area

        # ensure contour is almost a rectangle is
        try:
            corners = utils.find_corners(poly_contours[0])  # @IgnoreException
            if not utils.is_rect(*corners):
                logger.warning("Not a Rectangle")
                poly_contours = []
                corners = []
        except IndexError:
            corners = []

        # TODO: select contour with the least shift
        return poly_contours, corners

    def warp_sudoku_grid(self, frame, points):
        pts2 = np.float32([[0, 0], [self.warp_img_height-1, 0],
                           [0, self.warp_img_width-1], [self.warp_img_height - 1, self.warp_img_width-1]])
        logger.debug(pts2)
        matrix = cv2.getPerspectiveTransform(np.float32(points), pts2)
        result = cv2.warpPerspective(
            frame, matrix, (self.warp_img_width, self.warp_img_width))
        return result

    def find_sudoku_grid(self, img: np.ndarray):
        _, blur_frame = self.preprocess(img)
        blur_frame = self.find_straight_edges(blur_frame)

        contours, hierarchy = cv2.findContours(
            blur_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return None

        contour_utils = ContourUtil(contours, hierarchy)
        grids, grid_corners = self._find_sudoku_from_contours(
            contour_utils, img.shape[:2])

        if not grid_corners:
            return None

        top_left, top_right, bottom_right, bottom_left = grid_corners

        warped_frame = self.warp_sudoku_grid(
            img, [top_left, top_right, bottom_left, bottom_right])
        return warped_frame

    def run(self,):
        frame_num = 0
        print("I am here")
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            logger.debug(f"Frame number : {frame_num}")
            if not ret:
                break
            cv2.imshow("frame", frame)
            sudoku_img = self.find_sudoku_grid(frame)
            key = cv2.waitKey(1) & 0xff
            if sudoku_img is not None:
                cv2.imshow("sudoku image", sudoku_img)

                key = cv2.waitKey(0)

            if key == ord('y') or key == ord('Y'):
                number_img, number_contours = self.find_grid_cells(sudoku_img)
                if len(number_contours) < 16:
                    continue
                cv2.imshow("numbers", number_img)
                
                key = cv2.waitKey(0) & 0xff

                if key == ord("q") or key == ord('Q'):
                    break

            elif key == ord("q") or key == ord("Q"):
                break

            frame_num += 1

        self.cap.release()
        cv2.destroyAllWindows()
