import logging
import cv2
import numpy as np
from contour import ContourUtil
import math
from python_research import Color as C
from skimage.segmentation import clear_border
import grpc
import python_research.torchserve_proto.inference_pb2 as inference_pb2
import python_research.torchserve_proto.inference_pb2_grpc as inference_pb2_grpc
from python_research import Color
from tensorflow import make_tensor_proto
import torch
import requests


GUASSIAN_KERNEL_SIZE=(3, 3)
GUASSIAN_SIGMA = 10.0
CANNY_THRESHOLD_1 = 50
CANNY_THRESHOLD_2 = 100
CANNY_APERTURE_SIZE = 3

logging.basicConfig(level=logging.INFO)
channel = grpc.insecure_channel('localhost:8084')
inference_stub = inference_pb2_grpc.InferenceAPIsServiceStub(channel)
host = "localhost"
port = 8081

url = f"http://{host}:{port}/predictions/digitmodel"

session = requests.Session()

def predict_number(img):
    img = torch.from_numpy((img/255).astype(np.float32)).permute(2, 0, 1)
    img_as_bytes = make_tensor_proto(img).tensor_content
    input_data = {"data": img_as_bytes, "shape": str(tuple(img.shape)).encode('utf-8')}

    prediction = inference_stub.Predictions(inference_pb2.PredictionsRequest(model_name="digitmodel", input = input_data))

    return prediction

def predict_number_http(img):
    img = torch.from_numpy((img/255).astype(np.float32)).permute(2, 0, 1)
    img_as_bytes = make_tensor_proto(img).tensor_content
    data = {"data": img_as_bytes, "shape": str(tuple(img.shape)).encode('utf-8')}

    r = session.get(url, data = data)

    return r.text


def distance(p, q):
        return math.sqrt((p[0]-q[0])**2 + (p[1]-q[1])**2)

def img_area(img):
    return img.shape[0] * img.shape[1]

def find_contour_stats(contours):
    areas = np.array([cv2.contourArea(contour) for contour in contours])
    return areas, np.mean(areas), np.std(areas)

def find_grid_coords(contours, img_shape=(500, 500)):
        # img_shape - (height, width)
        row_division = np.linspace(0, img_shape[0], 10)
        column_division = np.linspace(0, img_shape[1], 10)

        xs = []
        ys = []
        for contour in contours:
            x, y, _, _ = cv2.boundingRect(contour)
            xs.append(x)
            ys.append(y)

        columns = np.digitize(xs, column_division, right=True)
        rows = np.digitize(ys, row_division, right=True)

        return list(zip(rows, columns))

def compute_bounding_diagonals(contours):
        distances = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            distances.append(distance((x, y), (x+w, y+h)))
        return distances

def draw_bounding_rects(img, contours, color=Color.WHITE.value):
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(img, (x, y), (x+w, y+h), color=color, thickness=1)
        return img

#find contours which have the same area.   
def find_grid_cells(img: np.ndarray):
        gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_area = gray_frame.shape[0] * gray_frame.shape[1]
        blur_frame = cv2.GaussianBlur(
            gray_frame,GUASSIAN_KERNEL_SIZE, GUASSIAN_SIGMA)
        edge_image = cv2.Canny(blur_frame, CANNY_THRESHOLD_1, CANNY_THRESHOLD_2,
                               apertureSize=CANNY_APERTURE_SIZE, L2gradient=True)

        kernel = np.ones((3, 3), dtype=np.uint8)
        edge_image = cv2.morphologyEx(edge_image, cv2.MORPH_OPEN, kernel, iterations=1)
        # edge_image = cv2.morphologyEx(edge_image, cv2.MORPH_DILATE, kernel, iterations=3)
        edge_image = cv2.morphologyEx(edge_image, cv2.MORPH_CLOSE, kernel, iterations=2)

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

        partition_condition = (
            contour_areas <= area_partition_value) & max_area_condition

        number_areas = contour_areas[partition_condition]
        number_area_mean = np.median(number_areas)
        number_area_std = np.std(number_areas)

        condition = (contour_areas <= (number_area_mean + (1 * number_area_std))
                     ) & (contour_areas >= (number_area_mean - (1 * number_area_std)))

        number_contours = child_contours[condition]

        number_contour_diagonals = np.array(
            compute_bounding_diagonals(number_contours))
        max_diagonal = 0.12 * distance((0, 0), (500, 500))

        diagonal_condition = number_contour_diagonals <= max_diagonal
        final_numbers = number_contours[diagonal_condition]

        # TODO: identify the coordinates where numbers are detected
        # detected contours - green
        cv2.drawContours(zero_img, contourIdx=-1,
                         contours=contours, color=Color.GREEN.value)
        # contours at bottom of hierarchy - blue
        cv2.drawContours(zero_img, contourIdx=-1,
                         contours=child_contours, color=Color.BLUE.value)

        # contours predicted as numers - red
        cv2.drawContours(zero_img, contourIdx=-1,
                         contours=final_numbers, color=Color.RED.value)

        draw_bounding_rects(zero_img, final_numbers)
        indexes = find_grid_coords(final_numbers, img.shape)
        return zero_img, final_numbers, indexes

def crop_img(img, top_left, bottom_right):
    x1,y1= top_left
    x2, y2 = bottom_right
    return img[y1:y2, x1:x2]

def draw_cirle(img, pt, color = C.WHITE.value, radius = 1, thickness=1):
    cv2.circle(img, pt[0], pt[1], radius = radius, color = color, thickness=thickness)

# https://stackoverflow.com/questions/35854197/how-to-use-opencvs-connectedcomponentswithstats-in-python
def is_number_cell(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    thresh_img = clear_border(thresh_img)

    min_area = img_area(thresh_img) *  0.05

    if (thresh_img==255).sum() >= min_area:
        return True
        
    return False

def identify_grid_cells(img: np.ndarray):
    img = cv2.GaussianBlur(
            img, GUASSIAN_KERNEL_SIZE, GUASSIAN_SIGMA)
    rows, columns = img.shape[:2]
    row_split = np.linspace(0, rows, 10)
    column_split = np.linspace(0, columns, 10)

    for row in range(9):
        row_top, row_bottom = map(round, row_split[row:row+2])
        for column in range(9):
            col_left, col_right = map(round, column_split[column:column+2])
            top_left = (col_left, row_top)
            bottom_right = (col_right, row_bottom)
            cropped_img = crop_img(img, top_left, bottom_right)
            if is_number_cell(cropped_img):
                pred = predict_number_http(cropped_img)
                logging.info(f"Cell: ({row+1}, {column+1}) - {pred}")
                #predict number


def find_grid_cells_v1(self, img: np.ndarray):
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
    img = cv2.imread("detected_grid.png")
    identify_grid_cells(img)
    