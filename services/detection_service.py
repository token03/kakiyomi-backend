from typing import List
from ultralytics import YOLO

from services.utils.pipeline_utils import does_rectangle_fit
from .utils.textblock import TextBlock, adjust_text_line_coordinates, sort_blk_list
import largestinteriorrectangle as lir
import numpy as np
import cv2

class DetectionService:
    def __init__(self):
        print("Initializing Detection Service")
        self.BUBBLE_MODEL_PATH = '../models/detection/comic-speech-bubble-detector.pt'
        self.TEXT_SEG_MODEL_PATH = '../models/detection/comic-text-segmenter.pt'
        self.TEXT_DETECT_MODEL_PATH = '../models/detection/manga-text-detector.pt'

        self.bubble_detection = YOLO(self.BUBBLE_MODEL_PATH)
        self.text_segmentation = YOLO(self.TEXT_SEG_MODEL_PATH)
        self.text_detection = YOLO(self.TEXT_DETECT_MODEL_PATH)

        self.device = 'cpu'  # TODO: add support for GPU
        print("Detection Service initialized")

    def clean_up(self):
        print("Cleaning up detection models")
        del self.bubble_detection
        del self.text_segmentation
        del self.text_detection
        print("Detection models cleaned up")

    def detect(self, img, source_lang=None) -> List[TextBlock]:
        h, w, _ = img.shape
        size = (h, w) if h >= w * 5 else 1024
        det_size = (h, w) if h >= w * 5 else 640

        bubble_detec_result = self.bubble_detection(img, device=self.device, imgsz=size, conf=0.1, verbose=False)[0]
        txt_seg_result = self.text_segmentation(img, device=self.device, imgsz=size, conf=0.1, verbose=False)[0]
        txt_detect_result = self.text_detection(img, device=self.device, imgsz=det_size, conf=0.2, verbose=False)[0]

        combined = self.combine_results(bubble_detec_result, txt_seg_result, txt_detect_result, img)

        blk_list = [TextBlock(txt_bbox, bble_bbox, txt_class, inp_bboxes)
                for txt_bbox, bble_bbox, inp_bboxes, txt_class in combined]

        # sort the blocks (rtl if japanese)  
        rtl = True if source_lang == 'Japanese' else False
        sorted_result = sort_blk_list(blk_list, rtl)

        return sorted_result

    def combine_results(self, bubble_detec_results, text_seg_results, text_detect_results, image):
        bubble_bounding_boxes = np.array(bubble_detec_results.boxes.xyxy.cpu(), dtype="int")
        seg_text_bounding_boxes = np.array(text_seg_results.boxes.xyxy.cpu(), dtype="int")
        detect_text_bounding_boxes = np.array(text_detect_results.boxes.xyxy.cpu(), dtype="int")

        text_bounding_boxes = self.merge_bounding_boxes(seg_text_bounding_boxes, detect_text_bounding_boxes)
        text_bounding_boxes = self.filter_bounding_boxes(text_bounding_boxes)

        text_blocks_bboxes = []
        for bbox in text_bounding_boxes:
            adjusted_bboxes = self.get_inpaint_bboxes(bbox, image)
            text_blocks_bboxes.append(adjusted_bboxes)

        raw_results = []
        text_matched = [False] * len(text_bounding_boxes)

        if text_bounding_boxes is not None or len(text_bounding_boxes) > 0:
            for txt_idx, txt_box in enumerate(text_bounding_boxes):
                for bble_box in bubble_bounding_boxes:
                    if does_rectangle_fit(bble_box, txt_box):
                        raw_results.append((txt_box, bble_box, text_blocks_bboxes[txt_idx], 'text_bubble'))
                        text_matched[txt_idx] = True
                        break
                    elif self.do_rectangles_overlap(bble_box, txt_box):
                        raw_results.append((txt_box, bble_box, text_blocks_bboxes[txt_idx], 'text_free'))
                        text_matched[txt_idx] = True
                        break
                if not text_matched[txt_idx]:
                    raw_results.append((txt_box, None, text_blocks_bboxes[txt_idx], 'text_free'))

        return raw_results

    def get_inpaint_bboxes(self, text_bbox, image):
        x1, y1, x2, y2 = adjust_text_line_coordinates(text_bbox, 0, 10, image)
        crop = image[y1:y2, x1:x2]
        content_bboxes = self.detect_content_in_bbox(crop)
        adjusted_bboxes = []
        for bbox in content_bboxes:
            lx1, ly1, lx2, ly2 = bbox
            adjusted_bbox = (x1 + lx1, y1 + ly1, x1 + lx2, y1 + ly2)
            adjusted_bboxes.append(adjusted_bbox)
        return adjusted_bboxes

    def calculate_iou(self, rect1, rect2) -> float:
        x1 = max(rect1[0], rect2[0])
        y1 = max(rect1[1], rect2[1])
        x2 = min(rect1[2], rect2[2])
        y2 = min(rect1[3], rect2[3])
        intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
        rect1_area = (rect1[2] - rect1[0]) * (rect1[3] - rect1[1])
        rect2_area = (rect2[2] - rect2[0]) * (rect2[3] - rect2[1])
        union_area = rect1_area + rect2_area - intersection_area
        iou = intersection_area / union_area if union_area != 0 else 0
        return iou

    def do_rectangles_overlap(self, rect1, rect2, iou_threshold: float = 0.2) -> bool:
        iou = self.calculate_iou(rect1, rect2)
        return iou >= iou_threshold

    def merge_boxes(self, box1, box2):
        return [
            min(box1[0], box2[0]),
            min(box1[1], box2[1]),
            max(box1[2], box2[2]),
            max(box1[3], box2[3])
        ]

    def merge_bounding_boxes(self, seg_boxes, detect_boxes):
        merged_boxes = []
        for seg_box in seg_boxes:
            running_box = seg_box
            for detect_box in detect_boxes:
                if does_rectangle_fit(running_box, detect_box):
                    continue
                if self.do_rectangles_overlap(running_box, detect_box, 0.02):
                    running_box = self.merge_boxes(running_box, detect_box)
            merged_boxes.append(running_box)
        for detect_box in detect_boxes:
            add_box = True
            for merged_box in merged_boxes:
                if self.do_rectangles_overlap(detect_box, merged_box, 0.1) or does_rectangle_fit(merged_box, detect_box):
                    add_box = False
                    break
            if add_box:
                merged_boxes.append(detect_box)
        final_boxes = []
        for i, box in enumerate(merged_boxes):
            running_box = box
            for j, other_box in enumerate(merged_boxes):
                if i == j:
                    continue
                if self.do_rectangles_overlap(running_box, other_box, 0.1) and not does_rectangle_fit(running_box, other_box):
                    running_box = self.merge_boxes(running_box, other_box)
            final_boxes.append(running_box)
        unique_boxes = []
        seen_boxes = []
        for box in final_boxes:
            duplicate = False
            for seen_box in seen_boxes:
                if (np.array_equal(box, seen_box) or self.do_rectangles_overlap(box, seen_box, 0.6)):
                    duplicate = True
                    break
            if not duplicate:
                unique_boxes.append(box)
                seen_boxes.append(box)
        return np.array(unique_boxes)

    def detect_content_in_bbox(self, image):
        if image is None or image.size == 0:
            return []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        binary_white_text = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        binary_black_text = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        num_labels_white, labels_white, stats_white, centroids_white = cv2.connectedComponentsWithStats(binary_white_text, connectivity=8)
        num_labels_black, labels_black, stats_black, centroids_black = cv2.connectedComponentsWithStats(binary_black_text, connectivity=8)
        min_area = 10
        content_bboxes = []
        height, width = image.shape[:2]
        for i in range(1, num_labels_white):
            area = stats_white[i, cv2.CC_STAT_AREA]
            if area > min_area:
                x1 = stats_white[i, cv2.CC_STAT_LEFT]
                y1 = stats_white[i, cv2.CC_STAT_TOP]
                w = stats_white[i, cv2.CC_STAT_WIDTH]
                h = stats_white[i, cv2.CC_STAT_HEIGHT]
                x2 = x1 + w
                y2 = y1 + h
                if x1 > 0 and y1 > 0 and x1 + w < width and y1 + h < height:
                    content_bboxes.append((x1, y1, x2, y2))
        for i in range(1, num_labels_black):
            area = stats_black[i, cv2.CC_STAT_AREA]
            if area > min_area:
                x1 = stats_black[i, cv2.CC_STAT_LEFT]
                y1 = stats_black[i, cv2.CC_STAT_TOP]
                w = stats_black[i, cv2.CC_STAT_WIDTH]
                h = stats_black[i, cv2.CC_STAT_HEIGHT]
                x2 = x1 + w
                y2 = y1 + h
                if x1 > 0 and y1 > 0 and x1 + w < width and y1 + h < height:
                    content_bboxes.append((x1, y1, x2, y2))
        return content_bboxes

    def filter_bounding_boxes(self, bboxes, width_tolerance=5, height_tolerance=5):
        def is_close(value1, value2, tolerance):
            return abs(value1 - value2) <= tolerance
        return [
            bbox for bbox in bboxes
            if not (is_close(bbox[0], bbox[2], width_tolerance) or is_close(bbox[1], bbox[3], height_tolerance))
        ]

    def adjust_contrast_brightness(self, img: np.ndarray, contrast: float = 1.0, brightness: int = 0):
        brightness += int(round(255 * (1 - contrast) / 2))
        return cv2.addWeighted(img, contrast, img, 0, brightness)

    def ensure_gray(self, img: np.ndarray):
        if len(img.shape) > 2:
            return cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
        return img.copy()

    def make_bubble_mask(self, frame: np.ndarray):
        image = frame.copy()
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        stage_1 = cv2.drawContours(np.zeros_like(image), contours, -1, (255, 255, 255), thickness=2)
        stage_1 = cv2.bitwise_not(stage_1)
        stage_1 = cv2.cvtColor(stage_1, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(stage_1, 200, 255, cv2.THRESH_BINARY)
        num_labels, labels = cv2.connectedComponents(binary_image)
        largest_island_label = np.argmax(np.bincount(labels.flat)[1:]) + 1
        mask = np.zeros_like(image)
        mask[labels == largest_island_label] = 255
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return self.adjust_contrast_brightness(mask, 100)

    def bubble_contour(self, frame_mask: np.ndarray):
        gray = self.ensure_gray(frame_mask)
        ret, thresh = cv2.threshold(gray, 200, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        largest_contour = max(contours, key=cv2.contourArea)
        return largest_contour

    def bubble_interior_bounds(self, frame_mask: np.ndarray):
        bble_contour = self.bubble_contour(frame_mask)
        polygon = np.array([bble_contour[:, 0, :]])
        rect = lir.lir(polygon)
        x1, y1 = lir.pt1(rect)
        x2, y2 = lir.pt2(rect)
        return x1, y1, x2, y2