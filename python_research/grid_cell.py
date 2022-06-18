import cv2
import numpy as np
from contour import ContourUtil

from python_research import Color

GUASSIAN_KERNEL_SIZE=(3, 3)
GUASSIAN_SIGMA = 10.0
CANNY_THRESHOLD_1 = 80
CANNY_THRESHOLD_2 = 130
CANNY_APERTURE_SIZE = 3


def img_area(img):
    return img.shape[0] * img.shape[1]

def find_contour_stats(contours):
    areas = np.array([cv2.contourArea(contour) for contour in contours])
    return areas, np.mean(areas), np.std(areas)

def find_partition_value(self, areas):
    
    pass

#find contours which have the same area.   

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
    contour_img = zero_img.copy()
    cv2.drawContours(contour_img, contourIdx=-1,
                        contours=child_contours, color=Color.MAGENTA.value)

    max_area_condition = contour_areas < (0.012 * img_area)
    max_area_img = zero_img.copy()
    cv2.drawContours(max_area_img, contourIdx=-1,
                        contours=child_contours[max_area_condition], color=Color.CYAN.value)

    sorted_areas = np.sort(
        contour_areas[max_area_condition])
    diff = np.diff(sorted_areas)
    area_partition_value = sorted_areas[np.argwhere(
        diff == diff.max())[0].item()]

    

    partition_condition = (contour_areas <= area_partition_value) & max_area_condition
    small_contour_img = zero_img.copy()
    cv2.drawContours(small_contour_img, contourIdx=-1,
                        contours=child_contours[partition_condition], color=Color.TEAL.value)

    number_areas = contour_areas[partition_condition]
    number_area_mean = np.median(number_areas)
    number_area_std = np.std(number_areas)

    condition = (contour_areas <= (number_area_mean + (1 * number_area_std))
                    ) & (contour_areas >= (number_area_mean - (1 * number_area_std)))

    number_contours = child_contours[condition]

    non_number_contours = child_contours[~condition]
    non_numb_img = zero_img.copy()
    cv2.drawContours(non_numb_img, contourIdx=-1,
                        contours=non_number_contours, color=Color.PURPLE.value)
    cv2.drawContours(non_numb_img, contourIdx=-1,
                        contours=number_contours, color=Color.RED.value)
    non_numb_img = self.draw_bounding_rects(non_numb_img, number_contours)

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

    # return zero_img, final_numbers

    cv2.imshow("Edge frame", edge_image)
    cv2.imshow("Image frame", zero_img)
    key = cv2.waitKey(0)
    if key == ord('q') & 0xff or key == ord("Q") & 0xff:
        return

    if key == ord('p') & 0xff or key == ord('P') & 0xff:
        cv2.waitKey(0)

if __name__ == "__main__":
    img = cv2.imread("python/detected_grid.png")
    find_grid_cells(img)
    