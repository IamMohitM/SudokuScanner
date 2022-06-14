import cv2
import numpy as np
from contour import ContourUtil

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

def find_grid_cells(img):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_frame = cv2.GaussianBlur(
            gray_frame, GUASSIAN_KERNEL_SIZE, GUASSIAN_SIGMA)
    edge_image = cv2.Canny(blur_frame, CANNY_THRESHOLD_1, CANNY_THRESHOLD_2,
                               apertureSize=CANNY_APERTURE_SIZE, L2gradient=True)

    edge_image = cv2.morphologyEx(edge_image, cv2.MORPH_CLOSE, np.ones((3, 3), dtype = np.uint8,), iterations=3)

    contours, hierarchy = cv2.findContours(
            edge_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    contour_util = ContourUtil(contours, hierarchy)
    contours = np.array(contours, dtype=object)
    hierarchy = np.squeeze(hierarchy)

    child_contours, _ = contour_util.find_contour_with_no_children()
    # no_child_condition = hierarchy[:, 2] == -1
    # child_ids = np.argwhere(no_child_condition).flatten()
    # child_contours = contours[child_ids]

    contour_areas, contour_area_mean, contour_area_std = find_contour_stats(child_contours)

    sorted_areas = np.sort(contour_areas[contour_areas < 0.013 * img_area(gray_frame)])
    diff = np.diff(sorted_areas)
    partition_value = sorted_areas[np.argwhere(diff == diff.max())[0].item()]

    condition = (contour_areas <= partition_value)
    number_contours = child_contours[condition]
    
    contour_areas, contour_area_mean, contour_area_std = find_contour_stats(number_contours)
    condition = (contour_areas <= (contour_area_mean + 1.5 * contour_area_std)) & (contour_areas >= (contour_area_mean - 1.5 * contour_area_std))
    number_contours = number_contours[condition]

    zero_img = np.zeros_like(img)
    cv2.drawContours(zero_img, contourIdx=-1,
                         contours=contours, color=(0, 255, 0))
    cv2.drawContours(zero_img, contourIdx=-1,
                         contours=child_contours, color=(255, 0, 0))
    cv2.drawContours(zero_img, contourIdx=-1,
                         contours=number_contours, color=(0, 0, 255))

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
    