# -*- coding: utf-8 -*-

import sys
import os
import cv2
import numpy as np
import mediapipe as mp
import datetime
import math
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton,
                             QVBoxLayout, QHBoxLayout, QGridLayout, QFileDialog,
                             QMessageBox, QInputDialog, QWidget, QGroupBox,
                             QSpinBox, QDoubleSpinBox, QComboBox, QTabWidget,
                             QFrame, QSplitter, QStatusBar, QProgressBar, QStyle)
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont, QIcon
from PyQt5.QtCore import Qt, QTimer, QPoint, QRect, QSize, pyqtSignal, QThread

# --- 人脸扶正功能模块 ---
def align_face_from_image(image: np.ndarray):
    """
    接收一张OpenCV格式的图像，检测人脸并根据双眼外眼角坐标进行扶正。
    此函数直接处理图像数据，而不是文件路径。

    Args:
        image (np.ndarray): 输入的BGR格式图像。

    Returns:
        numpy.ndarray: 扶正后的BGR格式图片。如果未检测到人脸，则返回 None。
    """
    if image is None:
        return None

    # 初始化 MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5) as face_mesh:

        # 将 BGR 图像转换为 RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)
        
        img_h, img_w, _ = image.shape

        if not results.multi_face_landmarks:
            print("警告: 在扶正步骤中未检测到人脸。")
            return None

        landmarks = results.multi_face_landmarks[0].landmark

        # 关键点索引: 左眼外眼角(33), 右眼外眼角(263)
        left_eye_corner = (int(landmarks[33].x * img_w), int(landmarks[33].y * img_h))
        right_eye_corner = (int(landmarks[263].x * img_w), int(landmarks[263].y * img_h))

        # 计算旋转角度
        dx = right_eye_corner[0] - left_eye_corner[0]
        dy = right_eye_corner[1] - left_eye_corner[1]
        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)

        print(f"检测到眼角连线角度: {angle_deg:.2f} 度，正在扶正...")

        # 执行旋转
        center = (img_w // 2, img_h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)

        # 计算新边界，防止裁剪
        cos = np.abs(rotation_matrix[0, 0])
        sin = np.abs(rotation_matrix[0, 1])
        new_w = int((img_h * sin) + (img_w * cos))
        new_h = int((img_h * cos) + (img_w * sin))

        # 调整旋转矩阵以居中
        rotation_matrix[0, 2] += (new_w / 2) - center[0]
        rotation_matrix[1, 2] += (new_h / 2) - center[1]

        # 应用旋转
        aligned_image = cv2.warpAffine(
            image,
            rotation_matrix,
            (new_w, new_h),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255) # 白色填充
        )
        
        return aligned_image

# --- 镜框检测功能模块 ---
def calculate_distance(p1, p2):
    """计算两点之间的欧几里得距离"""
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def detect_glasses_frames(image):
    """
    检测镜框并返回关键点。
    
    Args:
        image: OpenCV格式的图像数据
        
    Returns:
        Dictionary with detected points (global coordinates)
    """
    # MediaPipe关键点索引
    # 瞳孔
    LEFT_PUPIL_CENTER = 473
    RIGHT_PUPIL_CENTER = 468
    
    # 虹膜关键点 (虹膜周围的5个点)
    LEFT_IRIS_LANDMARKS = [473, 474, 475, 476, 477]
    RIGHT_IRIS_LANDMARKS = [468, 469, 470, 471, 472]
    
    # 眼角
    LEFT_EYE_INNER_CORNER = 362
    LEFT_EYE_OUTER_CORNER = 263
    RIGHT_EYE_INNER_CORNER = 133
    RIGHT_EYE_OUTER_CORNER = 33
    
    # 初始化 MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5) as face_mesh:
        
        if image is None:
            print("Error: Invalid image data")
            return None
            
        original_image = image.copy()
        image_height, image_width, _ = image.shape
        
        # 面部关键点检测
        rgb_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_image)
        
        if not results.multi_face_landmarks:
            print("Failed: Could not detect face landmarks or pupils.")
            return None
            
        # 提取面部关键点
        face_landmarks = results.multi_face_landmarks[0]  # 假设只有一张脸
        landmarks = face_landmarks.landmark
        
        # 初始化数据结构来存储关键点坐标
        eye_data = {
            'left': {
                'pupil_center': None,
                'iris_points': [],
                'inner_corner': None,
                'outer_corner': None,
                'iris_radius': 0,
                'frame_top': None,
                'frame_bottom': None
            },
            'right': {
                'pupil_center': None,
                'iris_points': [],
                'inner_corner': None,
                'outer_corner': None,
                'iris_radius': 0,
                'frame_top': None,
                'frame_bottom': None
            }
        }
        
        # 提取关键点坐标（保留全精度的浮点数）
        # 左眼
        left_pupil = landmarks[LEFT_PUPIL_CENTER]
        eye_data['left']['pupil_center'] = (left_pupil.x * image_width, left_pupil.y * image_height)
        
        for idx in LEFT_IRIS_LANDMARKS:
            lm = landmarks[idx]
            eye_data['left']['iris_points'].append((lm.x * image_width, lm.y * image_height))
            
        left_inner = landmarks[LEFT_EYE_INNER_CORNER]
        eye_data['left']['inner_corner'] = (left_inner.x * image_width, left_inner.y * image_height)
        
        left_outer = landmarks[LEFT_EYE_OUTER_CORNER]
        eye_data['left']['outer_corner'] = (left_outer.x * image_width, left_outer.y * image_height)
        
        # 右眼
        right_pupil = landmarks[RIGHT_PUPIL_CENTER]
        eye_data['right']['pupil_center'] = (right_pupil.x * image_width, right_pupil.y * image_height)
        
        for idx in RIGHT_IRIS_LANDMARKS:
            lm = landmarks[idx]
            eye_data['right']['iris_points'].append((lm.x * image_width, lm.y * image_height))
            
        right_inner = landmarks[RIGHT_EYE_INNER_CORNER]
        eye_data['right']['inner_corner'] = (right_inner.x * image_width, right_inner.y * image_height)
        
        right_outer = landmarks[RIGHT_EYE_OUTER_CORNER]
        eye_data['right']['outer_corner'] = (right_outer.x * image_width, right_outer.y * image_height)
        
        # 计算每只眼睛的虹膜半径
        for eye in ['left', 'right']:
            center = eye_data[eye]['pupil_center']
            iris_points = eye_data[eye]['iris_points']
            
            # 跳过中心点（第一个点）
            distances = []
            for i in range(1, len(iris_points)):
                point = iris_points[i]
                dist = calculate_distance(center, point)
                distances.append(dist)
            
            # 将虹膜半径设为平均距离
            if distances:
                eye_data[eye]['iris_radius'] = sum(distances) / len(distances)
        
        # 使用整个图像作为ROI
        x, y, w, h = 0, 0, image_width, image_height
        
        # 创建虹膜掩码
        pupil_mask = np.zeros((image_height, image_width), dtype=np.uint8)
        
        # 为虹膜区域绘制圆 - 扩大20%
        valid_eyes = []
        for eye in ['left', 'right']:
            center = eye_data[eye]['pupil_center']
            radius = eye_data[eye]['iris_radius']
            
            if center and radius > 0:
                valid_eyes.append(eye)
                # 扩大半径20%
                expanded_radius = int(radius * 1.2)
                cv2.circle(pupil_mask, (int(center[0]), int(center[1])), expanded_radius, 255, -1)
        
        if not valid_eyes:
            print("Error: No valid eyes with pupil centers and iris radius detected.")
            return None
        
        # 轮廓检测
        gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        blurred_image = cv2.GaussianBlur(gray_image, (7, 7), 0)
        edges_image = cv2.Canny(blurred_image, 50, 150)
        contours, _ = cv2.findContours(edges_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 过滤虹膜区域内的轮廓
        filtered_contours = []
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                if pupil_mask[cY, cX] == 0:
                    filtered_contours.append(contour)
        
        # 使用眼角约束寻找框架边界
        contour_map = np.zeros_like(gray_image)
        cv2.drawContours(contour_map, filtered_contours, -1, 255, 1)
        
        # 处理每只眼睛以找到框架边界
        for eye in valid_eyes:
            pupil_center = eye_data[eye]['pupil_center']
            inner_corner = eye_data[eye]['inner_corner']
            outer_corner = eye_data[eye]['outer_corner']
            
            if not (pupil_center and inner_corner and outer_corner):
                continue
                
            # 确定X方向的搜索范围（在眼角之间）
            min_x = min(inner_corner[0], outer_corner[0])
            max_x = max(inner_corner[0], outer_corner[0])
            
            # 沿着瞳孔的X位置在Y轴上搜索上下框架边界
            pupil_x, pupil_y = pupil_center
            
            # 仅在瞳孔X在眼角约束范围内时搜索
            if min_x <= pupil_x <= max_x:
                upper_intersection_point, lower_intersection_point = None, None
                
                # 向上搜索
                for y_up in range(int(pupil_y), 0, -1):
                    if contour_map[y_up, int(pupil_x)] > 0:
                        upper_intersection_point = (pupil_x, y_up)
                        break
                        
                # 向下搜索
                for y_down in range(int(pupil_y), image_height):
                    if contour_map[y_down, int(pupil_x)] > 0:
                        lower_intersection_point = (pupil_x, y_down)
                        break
        
                # 寻找并存储上框架边界点 - 修改：限制X坐标在眼角范围内
                if upper_intersection_point:
                    for contour in filtered_contours:
                        if cv2.pointPolygonTest(contour, (int(upper_intersection_point[0]), int(upper_intersection_point[1])), False) >= 0:
                            # 仅筛选在X范围内的轮廓点
                            valid_points = [pt for pt in contour if min_x <= pt[0][0] <= max_x]
                            if valid_points:
                                topmost_point = min(valid_points, key=lambda pt: pt[0][1])[0]
                                # 存储全局坐标
                                eye_data[eye]['frame_top'] = (topmost_point[0], topmost_point[1])
                                break
        
                # 寻找并存储下框架边界点 - 修改：限制X坐标在眼角范围内
                if lower_intersection_point:
                    for contour in filtered_contours:
                        if cv2.pointPolygonTest(contour, (int(lower_intersection_point[0]), int(lower_intersection_point[1])), False) >= 0:
                            # 仅筛选在X范围内的轮廓点
                            valid_points = [pt for pt in contour if min_x <= pt[0][0] <= max_x]
                            if valid_points:
                                bottommost_point = max(valid_points, key=lambda pt: pt[0][1])[0]
                                # 存储全局坐标
                                eye_data[eye]['frame_bottom'] = (bottommost_point[0], bottommost_point[1])
                                break
        
        # 返回框架点
        return eye_data

class FaceDetectionThread(QThread):
    """用于后台运行面部检测的线程"""
    detection_complete = pyqtSignal(tuple, tuple, np.ndarray, dict)
    progress_update = pyqtSignal(int)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, image_data): # 修改：接收图像数据而不是路径
        super().__init__()
        self.image_data = image_data
        
    def run(self):
        try:
            # 初始化MediaPipe Face Mesh
            mp_face_mesh = mp.solutions.face_mesh
            face_mesh = mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5
            )
            
            # 使用传入的图像数据
            image = self.image_data
            if image is None:
                self.error_occurred.emit("传入的图像数据无效")
                return
                
            self.progress_update.emit(20)
            
            # 转换为RGB并处理
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_image)
            
            self.progress_update.emit(40)
            
            if not results.multi_face_landmarks:
                self.error_occurred.emit("未检测到面部或瞳孔，请确保照片清晰且为完整的正面脸部")
                face_mesh.close()
                return
            
            landmarks = results.multi_face_landmarks[0]
            h, w = image.shape[:2]
            
            # MediaPipe关键点索引
            left_pupil_idx = 473
            right_pupil_idx = 468
            
            # 计算左眼瞳孔中心
            left_pupil_x = landmarks.landmark[left_pupil_idx].x * w
            left_pupil_y = landmarks.landmark[left_pupil_idx].y * h
            
            # 计算右眼瞳孔中心
            right_pupil_x = landmarks.landmark[right_pupil_idx].x * w
            right_pupil_y = landmarks.landmark[right_pupil_idx].y * h
            
            left_pupil = (int(left_pupil_x), int(left_pupil_y))
            right_pupil = (int(right_pupil_x), int(right_pupil_y))
            
            self.progress_update.emit(70)
            
            # 新增：尝试检测镜框
            frames_data = detect_glasses_frames(image)
            
            self.progress_update.emit(100)
            face_mesh.close()
            
            # 发送结果信号，包括镜框数据
            self.detection_complete.emit(left_pupil, right_pupil, image, frames_data)
            
        except Exception as e:
            self.error_occurred.emit(f"面部检测出错: {str(e)}")


class MagnifierView(QLabel):
    """放大镜视图控件"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(240, 240)
        self.setMaximumSize(300, 300)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: #222222; border: 2px solid #555555;")
        self.setText("放大镜预览")
        
        # 初始化放大镜参数
        self.magnification = 8
        self.zoom_size = 240
        
    def update_view(self, image, pos, marked_points=None, selected_index=-1, point_colors=None):
        """更新放大镜视图"""
        if image is None or pos is None:
            return
            
        h, w = image.shape[:2]
        x, y = pos.x(), pos.y()
        
        # 确保坐标在图像范围内
        if not (0 <= x < w and 0 <= y < h):
            return
        
        # 计算要放大的区域大小
        source_size = self.zoom_size // self.magnification
        half_size = source_size // 2
        
        # 计算源区域边界
        x1 = max(0, x - half_size)
        y1 = max(0, y - half_size)
        x2 = min(w, x1 + source_size)
        y2 = min(h, y1 + source_size)
        
        # 如果接近边界，调整另一端
        if x2 - x1 < source_size and x1 > 0:
            x1 = max(0, x2 - source_size)
        if y2 - y1 < source_size and y1 > 0:
            y1 = max(0, y2 - source_size)
        
        # 提取原始区域
        roi = image[y1:y2, x1:x2]
        
        if roi.size == 0:
            return
            
        # 放大区域
        magnified = cv2.resize(roi, (roi.shape[1] * self.magnification, roi.shape[0] * self.magnification), 
                               interpolation=cv2.INTER_NEAREST)
        
        # 创建放大镜图像
        mag_image = np.zeros((self.zoom_size, self.zoom_size, 3), dtype=np.uint8)
        
        crop_x, crop_y = 0, 0
        
        # 计算如何将放大的图像放入mag_image中
        if magnified.shape[0] < self.zoom_size or magnified.shape[1] < self.zoom_size:
            pad_y = (self.zoom_size - magnified.shape[0]) // 2
            pad_x = (self.zoom_size - magnified.shape[1]) // 2
            mag_image[pad_y:pad_y+magnified.shape[0], pad_x:pad_x+magnified.shape[1]] = magnified
            center_x = int((x - x1) * self.magnification + pad_x)
            center_y = int((y - y1) * self.magnification + pad_y)
        else:
            crop_y = (magnified.shape[0] - self.zoom_size) // 2
            crop_x = (magnified.shape[1] - self.zoom_size) // 2
            mag_image = magnified[crop_y:crop_y+self.zoom_size, crop_x:crop_x+self.zoom_size]
            center_x = int((x - x1) * self.magnification) - crop_x
            center_y = int((y - y1) * self.magnification) - crop_y
        
        # 确保中心坐标在放大窗口范围内
        center_x = max(0, min(self.zoom_size - 1, center_x))
        center_y = max(0, min(self.zoom_size - 1, center_y))
        
        # 绘制十字线
        cv2.line(mag_image, (0, center_y), (self.zoom_size, center_y), (0, 255, 0), 1)
        cv2.line(mag_image, (center_x, 0), (center_x, self.zoom_size), (0, 255, 0), 1)
        
        # 绘制中心点标记
        cv2.circle(mag_image, (center_x, center_y), 3, (255, 0, 0), 1)
        
        # 绘制已标记的点
        if marked_points and point_colors:
            for i, point in enumerate(marked_points):
                if i < len(point_colors):
                    px, py = point
                    if x1 <= px < x2 and y1 <= py < y2:
                        mag_px = int((px - x1) * self.magnification)
                        mag_py = int((py - y1) * self.magnification)
                        
                        if magnified.shape[0] < self.zoom_size:
                            mag_px += (self.zoom_size - magnified.shape[1]) // 2
                            mag_py += (self.zoom_size - magnified.shape[0]) // 2
                        else:
                            mag_px -= crop_x
                            mag_py -= crop_y
                        
                        if 0 <= mag_px < self.zoom_size and 0 <= mag_py < self.zoom_size:
                            color = point_colors[i]
                            if i == selected_index:
                                # 选中状态：更亮的颜色
                                bright_color = tuple(min(255, int(c * 1.5)) for c in color)
                                cv2.circle(mag_image, (mag_px, mag_py), 8, bright_color, -1)
                                cv2.circle(mag_image, (mag_px, mag_py), 9, (255, 255, 255), 2)
                            else:
                                cv2.circle(mag_image, (mag_px, mag_py), 6, color, -1)
                                cv2.circle(mag_image, (mag_px, mag_py), 7, (0, 0, 0), 1)
        
        # 显示坐标信息
        coord_text = f"({x}, {y})"
        cv2.putText(mag_image, coord_text, (5, self.zoom_size - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        # 将OpenCV图像转换为QImage并显示
        h_mag, w_mag, c_mag = mag_image.shape
        q_img = QImage(mag_image.data, w_mag, h_mag, c_mag * w_mag, QImage.Format_RGB888).rgbSwapped()
        self.setPixmap(QPixmap.fromImage(q_img))

class ImageView(QLabel):
    """图像显示控件，支持点击标记和交互"""
    point_clicked = pyqtSignal(QPoint)
    mouse_moved = pyqtSignal(QPoint)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(640, 480)
        self.setStyleSheet("background-color: #222222; border: 1px solid #555555;")
        self.setText("请打开图像文件")
        
        self.original_image = None
        self.display_pixmap = None
        self.scale_factor = 1.0
        self.offset = QPoint(0, 0)
        
        # 修改点位顺序：按照右上、左上、右下、左下的顺序（从观察者角度）
        self.marked_points = [None, None, None, None]  # 右上、左上、右下、左下
        self.selected_point_index = -1
        
        # 四个位置对应的颜色 (BGR格式) - 新的顺序
        self.point_colors = [
            (0, 255, 255),  # 右上 - 黄色
            (0, 255, 0),    # 左上 - 绿色
            (200, 100, 200), # 右下 - 浅紫色
            (150, 150, 240)  # 左下 - 淡红色
        ]
        
        self.left_pupil = None
        self.right_pupil = None
        
        # 保存原始的未修改图像（用于恢复）
        self.pristine_image = None
        
        self.setMouseTracking(True)
        
    def load_image(self, image):
        if image is None: return
        self.original_image = image.copy()
        # 保存一个原始的未修改版本
        if self.pristine_image is None:
            self.pristine_image = image.copy()
        self.update_display()
        
    def restore_original_image(self):
        """恢复到原始的未修改图像"""
        if self.pristine_image is not None:
            self.original_image = self.pristine_image.copy()
            self.update_display()
        
    def set_pupils(self, left_pupil, right_pupil):
        self.left_pupil = left_pupil
        self.right_pupil = right_pupil
        self.update_display()
        
    def smart_add_point(self, click_pos):
        """智能识别点击位置并添加到对应位置 - 从观察者视角"""
        if not self.left_pupil or not self.right_pupil:
            return -1
            
        img_x = int((click_pos.x() - self.offset.x()) / self.scale_factor)
        img_y = int((click_pos.y() - self.offset.y()) / self.scale_factor)
        
        if not (0 <= img_x < self.original_image.shape[1] and 0 <= img_y < self.original_image.shape[0]):
            return -1

        # 计算点击位置相对于瞳孔的位置
        left_pupil_x, left_pupil_y = self.left_pupil
        right_pupil_x, right_pupil_y = self.right_pupil
        
        # 判断是左眼区域还是右眼区域 - 从观察者视角
        mid_x = (left_pupil_x + right_pupil_x) / 2
        is_right_side = img_x < mid_x  # 从观察者视角看，左边实际是右眼
        
        # 判断是上方还是下方
        target_pupil_y = right_pupil_y if is_right_side else left_pupil_y
        is_above = img_y < target_pupil_y
        
        # 确定目标位置索引 - 从观察者视角
        if is_right_side and is_above:
            target_index = 0  # 右上
        elif not is_right_side and is_above:
            target_index = 1  # 左上
        elif is_right_side and not is_above:
            target_index = 2  # 右下
        else:
            target_index = 3  # 左下
        
        # 添加点到对应位置
        self.marked_points[target_index] = (img_x, img_y)
        self.selected_point_index = target_index
        self.update_display()
        
        return target_index

    def select_point(self, index):
        if 0 <= index < 4 and self.marked_points[index] is not None:
            self.selected_point_index = index
            self.update_display()
            
    def move_selected_point(self, dx, dy):
        if 0 <= self.selected_point_index < 4 and self.marked_points[self.selected_point_index] is not None:
            x, y = self.marked_points[self.selected_point_index]
            new_x = max(0, min(self.original_image.shape[1] - 1, x + dx))
            new_y = max(0, min(self.original_image.shape[0] - 1, y + dy))
            self.marked_points[self.selected_point_index] = (new_x, new_y)
            self.update_display()
            return (new_x, new_y)  # 返回新位置用于更新放大镜
        return None
    
    def get_valid_points(self):
        """获取已标记的有效点"""
        return [p for p in self.marked_points if p is not None]
    
    def clear_all_points(self):
        """清除所有标记点"""
        self.marked_points = [None, None, None, None]
        self.selected_point_index = -1
        self.update_display()
    
    def update_display(self):
        if self.original_image is None: return
            
        display = self.original_image.copy()
        
        # 绘制瞳孔
        if self.left_pupil:
            cv2.circle(display, self.left_pupil, 6, (255, 0, 0), -1)
            cv2.circle(display, self.left_pupil, 8, (255, 255, 0), 2)
        if self.right_pupil:
            cv2.circle(display, self.right_pupil, 6, (255, 0, 0), -1)
            cv2.circle(display, self.right_pupil, 8, (255, 255, 0), 2)
        
        # 绘制标记点
        for i, point in enumerate(self.marked_points):
            if point is not None:
                color = self.point_colors[i]
                if i == self.selected_point_index:
                    # 选中状态：更大的圆圈和白色边框
                    cv2.circle(display, point, 8, color, -1)
                    cv2.circle(display, point, 10, (255, 255, 255), 2)
                else:
                    cv2.circle(display, point, 6, color, -1)
                    cv2.circle(display, point, 8, (0, 0, 0), 1)
        
        # 绘制连接线 - 右侧连接和左侧连接 
        if self.marked_points[0] and self.marked_points[2]:  # 右上与右下连接
            cv2.line(display, self.marked_points[0], self.marked_points[2], (0, 255, 0), 2)
        if self.marked_points[1] and self.marked_points[3]:  # 左上与左下连接
            cv2.line(display, self.marked_points[1], self.marked_points[3], (0, 255, 0), 2)
        
        h, w, c = display.shape
        q_img = QImage(display.data, w, h, c * w, QImage.Format_RGB888).rgbSwapped()
        self.display_pixmap = QPixmap.fromImage(q_img)
        
        self.update() # 触发 paintEvent
            
    def paintEvent(self, event):
        if not self.display_pixmap:
            super().paintEvent(event)
            return
        
        painter = QPainter(self)
        
        # 计算缩放和偏移
        pixmap_size = self.display_pixmap.size()
        widget_size = self.size()
        
        scale_w = widget_size.width() / pixmap_size.width()
        scale_h = widget_size.height() / pixmap_size.height()
        self.scale_factor = min(scale_w, scale_h)
        
        scaled_w = int(pixmap_size.width() * self.scale_factor)
        scaled_h = int(pixmap_size.height() * self.scale_factor)
        
        self.offset.setX((widget_size.width() - scaled_w) // 2)
        self.offset.setY((widget_size.height() - scaled_h) // 2)
        
        target_rect = QRect(self.offset, QSize(scaled_w, scaled_h))
        painter.drawPixmap(target_rect, self.display_pixmap)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.original_image is not None:
            self.point_clicked.emit(event.pos())
            
    def mouseMoveEvent(self, event):
        if self.original_image is not None:
            self.mouse_moved.emit(event.pos())
            
    def view_to_image_pos(self, view_pos):
        if self.scale_factor == 0: return (0, 0)
        x = int((view_pos.x() - self.offset.x()) / self.scale_factor)
        y = int((view_pos.y() - self.offset.y()) / self.scale_factor)
        return (x, y)

class ResultsView(QWidget):
    """测量结果显示控件"""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout()
        result_group = QGroupBox("测量结果")
        result_layout = QGridLayout()
        
        # 创建标签和值
        self.labels = {}
        fields = ["镜片高度:", "左眼瞳高:", "右眼瞳高:", "平均瞳高:", "瞳高差异:", "状态:"]
        for i, text in enumerate(fields):
            label = QLabel(text)
            value = QLabel("--")
            result_layout.addWidget(label, i, 0)
            result_layout.addWidget(value, i, 1)
            self.labels[text.split(':')[0]] = value
        
        result_group.setLayout(result_layout)
        layout.addWidget(result_group)
        
        self.save_button = QPushButton("导出报告")
        self.save_button.setEnabled(False)
        layout.addWidget(self.save_button)
        
        self.setLayout(layout)
        
    def update_results(self, lens_height, left_height, right_height):
        avg_height = (left_height + right_height) / 2
        diff = abs(left_height - right_height)
        
        self.labels["镜片高度"].setText(f"{lens_height:.2f} mm")
        self.labels["左眼瞳高"].setText(f"{left_height:.2f} mm")
        self.labels["右眼瞳高"].setText(f"{right_height:.2f} mm")
        self.labels["平均瞳高"].setText(f"{avg_height:.2f} mm")
        self.labels["瞳高差异"].setText(f"{diff:.2f} mm")
        
        if diff < 0.5:
            status, color = "双眼瞳高完全对称 ✓✓", "color: #4CAF50; font-weight: bold;"
        elif diff < 1.0:
            status, color = "双眼瞳高基本对称 ✓", "color: #8BC34A;"
        elif diff < 2.0:
            status, color = "双眼瞳高略有差异 !", "color: #FFC107;"
        else:
            status, color = "双眼瞳高差异较大 !!", "color: #F44336; font-weight: bold;"
            
        self.labels["状态"].setText(status)
        self.labels["状态"].setStyleSheet(color)
        
        self.save_button.setEnabled(True)
        
        return {
            "lens_height": lens_height, "left_height": left_height,
            "right_height": right_height, "avg_height": avg_height,
            "diff": diff, "status": status
        }
    
    def reset(self):
        for label in self.labels.values():
            label.setText("--")
            label.setStyleSheet("")
        self.save_button.setEnabled(False)

class PupilHeightCalculator(QMainWindow):
    """瞳高测量程序主窗口"""
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("瞳高测量程序 - 优化版 (集成镜框检测)")
        self.setWindowIcon(QIcon.fromTheme("camera-photo"))
        self.setGeometry(100, 100, 1400, 800)
        
        # --- 创建控件 ---
        central_widget = QWidget()
        main_layout = QHBoxLayout(central_widget)
        
        # 左侧面板 - 增大面积
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setFixedWidth(420)  # 从320增加到420

        # 右侧内容区 (图像和结果)
        right_splitter = QSplitter(Qt.Vertical)
        self.image_view = ImageView()
        self.results_view = ResultsView()
        right_splitter.addWidget(self.image_view)
        right_splitter.addWidget(self.results_view)
        right_splitter.setSizes([600, 200])

        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_splitter, 1)

        # --- 填充左侧面板 ---
        # 1. 操作控制
        control_group = QGroupBox("操作控制")
        control_layout = QVBoxLayout()
        self.open_button = QPushButton("1. 打开图像")
        self.open_button.setIcon(self.style().standardIcon(QStyle.SP_DialogOpenButton))
        self.detect_button = QPushButton("2. 检测瞳孔和镜框")
        self.detect_button.setEnabled(False)
        lens_layout = QHBoxLayout()
        lens_layout.addWidget(QLabel("镜片物理高度 (mm):"))
        self.lens_height_input = QDoubleSpinBox()
        self.lens_height_input.setRange(10, 80); self.lens_height_input.setValue(40);
        self.lens_height_input.setDecimals(1); self.lens_height_input.setSingleStep(0.5);
        lens_layout.addWidget(self.lens_height_input)
        self.calculate_button = QPushButton("4. 计算瞳高")
        self.calculate_button.setEnabled(False)
        self.clear_button = QPushButton("清除标记")
        self.clear_button.setEnabled(False)
        for w in [self.open_button, self.detect_button, self.calculate_button, self.clear_button]: 
            w.setMinimumHeight(35)
        control_layout.addWidget(self.open_button)
        control_layout.addWidget(self.detect_button)
        control_layout.addLayout(lens_layout)
        control_layout.addWidget(self.calculate_button)
        control_layout.addWidget(self.clear_button)
        control_group.setLayout(control_layout)
        left_layout.addWidget(control_group)

        # 2. 标记点控制 - 修改为新的顺序[右上 左上][右下 左下]
        marking_group = QGroupBox("3. 标记与微调")
        marking_layout = QVBoxLayout()
        
        # 颜色标记按钮
        self.point_buttons = []
        point_grid = QGridLayout()
        
        # 定义颜色和位置 - 新的顺序：[右上 左上][右下 左下]
        color_configs = [
            ("右上", "#FFFF00"),  # 黄色
            ("左上", "#00FF00"),  # 绿色
            ("右下", "#C864C8"),  # 浅紫色
            ("左下", "#F0A0A0")   # 淡红色
        ]
        
        # 创建2x2网格布局，新的顺序: [右上 左上][右下 左下]
        positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
        
        for i, (pos_name, color_hex) in enumerate(color_configs):
            button = QPushButton()
            button.setCheckable(True)
            button.setEnabled(False)  # 初始禁用
            button.setMinimumHeight(40)
            button.setMinimumWidth(80)
            button.setStyleSheet(f"""
                QPushButton {{
                    background-color: {color_hex};
                    border: 1px solid #CCCCCC;
                    border-radius: 5px;
                    font-weight: bold;
                    color: #000000;
                }}
                QPushButton:checked {{
                    border: 3px solid #000000;
                    background-color: {color_hex};
                }}
                QPushButton:disabled {{
                    background-color: #666666;
                    color: #999999;
                }}
            """)
            button.setText(pos_name)
            self.point_buttons.append(button)
            row, col = positions[i]
            point_grid.addWidget(button, row, col)
        
        marking_layout.addLayout(point_grid)

        # 微调按钮
        adj_layout = QGridLayout()
        self.up_button = QPushButton("↑"); self.down_button = QPushButton("↓")
        self.left_button = QPushButton("←"); self.right_button = QPushButton("→")
        self.adj_buttons = [self.up_button, self.down_button, self.left_button, self.right_button]
        adj_layout.addWidget(self.up_button, 0, 1); adj_layout.addWidget(self.down_button, 2, 1)
        adj_layout.addWidget(self.left_button, 1, 0); adj_layout.addWidget(self.right_button, 1, 2)
        for btn in self.adj_buttons: 
            btn.setEnabled(False)
            btn.setMinimumHeight(30)
        marking_layout.addLayout(adj_layout)
        marking_group.setLayout(marking_layout)
        left_layout.addWidget(marking_group)

        # 3. 放大镜
        self.magnifier = MagnifierView()
        left_layout.addWidget(self.magnifier)
        
        # 4. 说明
        instruction_group = QGroupBox("操作说明")
        instruction_layout = QVBoxLayout()
        instructions = [
            "· 照片要求: 相机与双眼平行，正视前方。",
            "· 程序会自动尝试扶正图像。",
            "· 程序会尝试自动检测镜框，如未成功需手动标记。",  # 更新
            "· 智能标记: 点击图像自动识别位置。",
            "· 颜色对应: 黄色(右上) 绿色(左上) 紫色(右下) 淡红色(左下)。",  # 更新说明
            "· 点击颜色按钮可选择对应标记点。",
            "· 使用方向键对选中点进行微调。"
        ]
        for instruction in instructions: 
            instruction_label = QLabel(instruction)
            instruction_label.setWordWrap(True)  # 启用自动换行
            instruction_layout.addWidget(instruction_label)
        instruction_group.setLayout(instruction_layout)
        left_layout.addWidget(instruction_group)
        left_layout.addStretch()

        self.setCentralWidget(central_widget)
        
        # --- 状态栏和进度条 ---
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)
        
        # --- 连接信号和槽 ---
        self.connect_signals()
        
        # --- 初始化变量 ---
        self.image_path = None
        self.detection_thread = None
        self.measurement_results = None
        self.frames_data = None  # 新增：存储镜框检测数据
        
    def connect_signals(self):
        self.open_button.clicked.connect(self.open_image)
        self.detect_button.clicked.connect(self.detect_pupils)
        self.clear_button.clicked.connect(self.clear_marks)
        self.calculate_button.clicked.connect(self.calculate_pupil_height)
        
        for i, button in enumerate(self.point_buttons):
            button.clicked.connect(lambda checked, idx=i: self.select_point(idx))
            
        self.up_button.clicked.connect(lambda: self.move_selected_point(0, -1))
        self.down_button.clicked.connect(lambda: self.move_selected_point(0, 1))
        self.left_button.clicked.connect(lambda: self.move_selected_point(-1, 0))
        self.right_button.clicked.connect(lambda: self.move_selected_point(1, 0))
        
        self.image_view.point_clicked.connect(self.on_image_clicked)
        self.image_view.mouse_moved.connect(self.on_mouse_moved)
        self.results_view.save_button.clicked.connect(self.save_results)
        
    def open_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "选择图像文件", "", "Images (*.jpg *.jpeg *.png *.bmp)")
        if file_path:
            self.image_path = file_path
            image = cv2.imread(self.image_path)
            if image is None:
                QMessageBox.critical(self, "错误", "无法加载图像文件。")
                return

            self.status_bar.showMessage("正在尝试扶正图像...")
            QApplication.processEvents() # 刷新UI显示状态信息

            # --- 新增：调用人脸扶正 ---
            aligned_image = align_face_from_image(image)
            
            final_image = None
            if aligned_image is not None:
                final_image = aligned_image
                status_msg = f"图像已扶正。已加载: {os.path.basename(self.image_path)}"
            else:
                final_image = image
                status_msg = f"扶正失败或未检测到人脸，使用原始图像。已加载: {os.path.basename(self.image_path)}"
            
            # 使用扶正后（或原始）的图像进行后续操作
            self.reset_state()
            self.image_view.load_image(final_image)
            self.status_bar.showMessage(status_msg)
            self.detect_button.setEnabled(True)
            self.clear_button.setEnabled(True)

    def reset_state(self):
        """重置程序状态以便加载新图片"""
        self.image_view.pristine_image = None # 关键：确保pristine_image被重置
        self.image_view.load_image(None)
        self.image_view.clear_all_points()
        self.image_view.left_pupil = None
        self.image_view.right_pupil = None
        self.image_view.update_display()

        self.results_view.reset()
        self.measurement_results = None
        self.frames_data = None  # 重置镜框数据
        self.calculate_button.setEnabled(False)
        self.detect_button.setEnabled(False)
        for btn in self.point_buttons: 
            btn.setEnabled(False)
            btn.setChecked(False)
        for btn in self.adj_buttons: 
            btn.setEnabled(False)

    def detect_pupils(self):
        if self.image_view.pristine_image is None: return
        self.status_bar.showMessage("正在检测瞳孔和镜框...")
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.open_button.setEnabled(False)
        self.detect_button.setEnabled(False)
        
        # 修改：将图像数据传递给线程
        self.detection_thread = FaceDetectionThread(self.image_view.pristine_image)
        self.detection_thread.detection_complete.connect(self.on_detection_complete)
        self.detection_thread.progress_update.connect(self.progress_bar.setValue)
        self.detection_thread.error_occurred.connect(self.on_detection_error)
        self.detection_thread.finished.connect(self.on_detection_finished)
        self.detection_thread.start()
        
    def on_detection_complete(self, left_pupil, right_pupil, image, frames_data):
        # 在检测瞳孔前，确保图像恢复到原始状态，清除之前的标记和计算结果
        self.image_view.restore_original_image()
        self.image_view.clear_all_points()
        self.results_view.reset()
        
        self.image_view.set_pupils(left_pupil, right_pupil)
        self.frames_data = frames_data
        
        if self.frames_data and all(self.frames_data['left'].get('frame_top') and 
                                    self.frames_data['left'].get('frame_bottom') and
                                    self.frames_data['right'].get('frame_top') and
                                    self.frames_data['right'].get('frame_bottom')):
            # 自动填充检测到的镜框点位
            self.auto_fill_frame_points()
            status_msg = "镜框已自动检测。如需调整，请点击颜色按钮选中点位后微调。"
        else:
            # 如果镜框检测不完整，提示用户手动标记
            missing_points = []
            if self.frames_data:
                if not self.frames_data['right'].get('frame_top'):
                    missing_points.append("右上")
                if not self.frames_data['left'].get('frame_top'):
                    missing_points.append("左上")
                if not self.frames_data['right'].get('frame_bottom'):
                    missing_points.append("右下")
                if not self.frames_data['left'].get('frame_bottom'):
                    missing_points.append("左下")
                
                # 填充已检测到的点
                self.auto_fill_frame_points(partial=True)
                
                if missing_points:
                    missing_str = ", ".join(missing_points)
                    QMessageBox.information(self, "部分检测成功", 
                                           f"自动检测未能识别所有镜框点，缺少: {missing_str}\n请手动添加缺失的点位。")
                    status_msg = f"请手动点击添加缺失的点位: {missing_str}"
                else:
                    status_msg = "镜框点位已自动检测，请检查并调整位置。"
            else:
                QMessageBox.information(self, "未检测到镜框", 
                                       "自动检测未能识别镜框，请手动标记四个点。")
                status_msg = "请点击图像标记镜片边缘。"
                
        self.enable_ui_after_detection()
        self.status_bar.showMessage(status_msg)
        
    def auto_fill_frame_points(self, partial=False):
        """自动填充检测到的镜框点位"""
        if not self.frames_data:
            return
            
        # 从观察者角度，右上、左上、右下、左下
        if self.frames_data['right'].get('frame_top') and (not partial or self.marked_points[0] is None):
            point = self.frames_data['right']['frame_top']
            self.image_view.marked_points[0] = (int(point[0]), int(point[1]))
            
        if self.frames_data['left'].get('frame_top') and (not partial or self.marked_points[1] is None):
            point = self.frames_data['left']['frame_top']
            self.image_view.marked_points[1] = (int(point[0]), int(point[1]))
            
        if self.frames_data['right'].get('frame_bottom') and (not partial or self.marked_points[2] is None):
            point = self.frames_data['right']['frame_bottom']
            self.image_view.marked_points[2] = (int(point[0]), int(point[1]))
            
        if self.frames_data['left'].get('frame_bottom') and (not partial or self.marked_points[3] is None):
            point = self.frames_data['left']['frame_bottom']
            self.image_view.marked_points[3] = (int(point[0]), int(point[1]))
            
        self.image_view.update_display()
        self.update_ui_state()
    
    def enable_ui_after_detection(self):
        """在检测完成后启用相关UI元素"""
        for button in self.point_buttons:
            button.setEnabled(True)
        
        valid_points = self.image_view.get_valid_points()
        self.calculate_button.setEnabled(len(valid_points) >= 4)
        
    def on_detection_error(self, error_msg):
        QMessageBox.warning(self, "检测错误", error_msg)
        self.status_bar.showMessage("瞳孔检测失败。")
        
    def on_detection_finished(self):
        self.progress_bar.setVisible(False)
        self.open_button.setEnabled(True)
        self.detect_button.setEnabled(True)
        
    def on_image_clicked(self, pos):
        if not self.image_view.left_pupil:
             QMessageBox.information(self, "提示", "请先进行瞳孔检测。")
             return
        
        # 智能添加点
        added_index = self.image_view.smart_add_point(pos)
        if added_index >= 0:
            # 启用标记与微调功能
            for button in self.point_buttons:
                button.setEnabled(True)
            self.update_ui_state()
            point = self.image_view.marked_points[added_index]
            
            # 点击后立刻更新放大镜视图
            img_pos = self.image_view.view_to_image_pos(pos)
            self.magnifier.update_view(
                self.image_view.original_image,
                QPoint(point[0], point[1]),  # 使用标记点的位置而不是鼠标位置
                self.image_view.get_valid_points(),
                added_index,  # 选中新添加的点
                self.image_view.point_colors
            )
            
            # 使用新的位置名称: [右上 左上][右下 左下]
            position_names = ["右上", "左上", "右下", "左下"]
            self.status_bar.showMessage(f"已标记{position_names[added_index]}位置: ({point[0]}, {point[1]})")
        
    def on_mouse_moved(self, pos):
        if self.image_view.original_image is None: return
        img_pos = self.image_view.view_to_image_pos(pos)
        self.magnifier.update_view(
            self.image_view.original_image, 
            QPoint(img_pos[0], img_pos[1]),
            self.image_view.get_valid_points(), 
            self.image_view.selected_point_index,
            self.image_view.point_colors
        )
        
    def select_point(self, index):
        # 清除其他按钮的选中状态
        for i, button in enumerate(self.point_buttons):
            if i != index: 
                button.setChecked(False)

        # 如果该位置有标记点，则选中它
        if self.image_view.marked_points[index] is not None:
            self.image_view.select_point(index)
            self.point_buttons[index].setChecked(True)
            for btn in self.adj_buttons: 
                btn.setEnabled(True)
            point = self.image_view.marked_points[index]
            
            # 在选择点后立刻更新放大镜
            self.magnifier.update_view(
                self.image_view.original_image,
                QPoint(point[0], point[1]),
                self.image_view.get_valid_points(),
                index,
                self.image_view.point_colors
            )
            
            # 使用新的位置名称: [右上 左上][右下 左下]
            position_names = ["右上", "左上", "右下", "左下"]
            self.status_bar.showMessage(f"已选择{position_names[index]}位置: ({point[0]}, {point[1]})")
        else:
            # 如果该位置没有标记点，取消选中
            self.image_view.selected_point_index = -1
            self.image_view.update_display()
            self.point_buttons[index].setChecked(False)
            for btn in self.adj_buttons: 
                btn.setEnabled(False)
            
    def move_selected_point(self, dx, dy):
        # 移动点并获取新的位置
        new_pos = self.image_view.move_selected_point(dx, dy)
        if new_pos:
            idx = self.image_view.selected_point_index
            # 使用新的位置名称: [右上 左上][右下 左下]
            position_names = ["右上", "左上", "右下", "左下"]
            self.status_bar.showMessage(f"移动{position_names[idx]}位置至: ({new_pos[0]}, {new_pos[1]})")
            
            # 在移动点后立刻更新放大镜
            self.magnifier.update_view(
                self.image_view.original_image,
                QPoint(new_pos[0], new_pos[1]),
                self.image_view.get_valid_points(),
                idx,
                self.image_view.point_colors
            )
            
    def clear_marks(self):
        # 检查是否有先前的瞳高数据和辅助标记
        if self.measurement_results is not None:
            # 恢复到原始图像（清除计算结果的可视化）
            self.image_view.restore_original_image()
            self.results_view.reset()
            self.measurement_results = None
            
        # 清除所有标记点
        self.image_view.clear_all_points()
        self.update_ui_state()
        # 清除标记后禁用标记按钮
        for btn in self.point_buttons:
            btn.setEnabled(False)
            btn.setChecked(False)
        self.status_bar.showMessage("已清除所有标记点。")
        
    def update_ui_state(self):
        valid_points = self.image_view.get_valid_points()
        num_points = len(valid_points)
        
        # 更新按钮选中状态
        for i, button in enumerate(self.point_buttons):
            has_point = self.image_view.marked_points[i] is not None
            button.setChecked(i == self.image_view.selected_point_index and has_point)
        
        # 只有4个点都标记完才能计算
        self.calculate_button.setEnabled(num_points >= 4)
        
        # 更新微调按钮状态
        if self.image_view.selected_point_index != -1:
             for btn in self.adj_buttons: 
                  btn.setEnabled(True)
        else:
             for btn in self.adj_buttons: 
                  btn.setEnabled(False)

        if self.image_view.left_pupil:
            self.status_bar.showMessage(f"已标记 {num_points}/4 个点")
        
    def calculate_pupil_height(self):
        valid_points = self.image_view.get_valid_points()
        if len(valid_points) < 4 or not self.image_view.left_pupil:
            QMessageBox.warning(self, "无法计算", "请确保已检测瞳孔并标记全部4个镜片边缘点。")
            return
            
        # 检查是否有先前的瞳高数据和辅助标记
        if self.measurement_results is not None:
            # 恢复到原始图像（清除先前的计算结果）
            self.image_view.restore_original_image()
            
        lens_height_mm = self.lens_height_input.value()
        
        # 获取四个标记点 - 使用新的顺序: [右上 左上][右下 左下]
        right_top = self.image_view.marked_points[0]      # 右上
        left_top = self.image_view.marked_points[1]       # 左上
        right_bottom = self.image_view.marked_points[2]   # 右下
        left_bottom = self.image_view.marked_points[3]    # 左下
        
        if None in [right_top, left_top, right_bottom, left_bottom]:
            QMessageBox.warning(self, "无法计算", "请确保标记了所有4个位置的点。")
            return
        
        left_lens_h_px = abs(left_top[1] - left_bottom[1])
        right_lens_h_px = abs(right_top[1] - right_bottom[1])
        
        if left_lens_h_px == 0 or right_lens_h_px == 0:
            QMessageBox.critical(self, "计算错误", "镜片标记点在垂直方向上重合，无法计算高度。")
            return

        avg_lens_h_px = (left_lens_h_px + right_lens_h_px) / 2.0
        scale = lens_height_mm / avg_lens_h_px
        
        # 计算瞳高：从下边缘点到瞳孔中心的距离
        left_pupil_h_px = abs(left_bottom[1] - self.image_view.left_pupil[1])
        right_pupil_h_px = abs(right_bottom[1] - self.image_view.right_pupil[1])
        
        left_pupil_h_mm = left_pupil_h_px * scale
        right_pupil_h_mm = right_pupil_h_px * scale
        
        self.measurement_results = self.results_view.update_results(lens_height_mm, left_pupil_h_mm, right_pupil_h_mm)
        self.visualize_results(left_pupil_h_mm, right_pupil_h_mm)
        
    def visualize_results(self, left_height, right_height):
        result_image = self.image_view.original_image.copy()
        
        # 获取标记点 - 使用新的顺序: [右上 左上][右下 左下]
        left_bottom = self.image_view.marked_points[3]  # 左下
        right_bottom = self.image_view.marked_points[2] # 右下
        left_pupil, right_pupil = self.image_view.left_pupil, self.image_view.right_pupil
        
        # 绘制瞳高线和文字
        for pupil, bottom, height, side in [(left_pupil, left_bottom, left_height, "L"), (right_pupil, right_bottom, right_height, "R")]:
            cv2.line(result_image, (pupil[0], pupil[1]), (pupil[0], bottom[1]), (0, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(result_image, pupil, 6, (0, 0, 255), -1)
            text = f"{side}: {height:.2f}mm"
            cv2.putText(result_image, text, (pupil[0] + 15, pupil[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # 更新显示
        self.image_view.load_image(result_image)
        
        # 保存结果图像
        default_name = os.path.splitext(os.path.basename(self.image_path))[0] + "_result.jpg"
        save_path, _ = QFileDialog.getSaveFileName(self, "保存结果图像", default_name, "JPEG Image (*.jpg);;PNG Image (*.png)")
        
        if save_path:
            try:
                # 注意：OpenCV保存的是BGR格式
                cv2.imwrite(save_path, result_image)
                self.status_bar.showMessage(f"计算完成。结果图像已保存至: {save_path}")
            except Exception as e:
                QMessageBox.critical(self, "保存失败", f"无法保存图像文件: {e}")
        else:
            self.status_bar.showMessage("计算完成。结果图像未保存。")

    def save_results(self):
        if self.measurement_results is None: return
        
        default_name = os.path.splitext(os.path.basename(self.image_path))[0] + "_report.txt"
        save_path, _ = QFileDialog.getSaveFileName(self, "保存测量报告", default_name, "Text Files (*.txt)")
        
        if save_path:
            try:
                with open(save_path, "w", encoding="utf-8") as f:
                    f.write("瞳高测量结果报告\n")
                    f.write("="*40 + "\n")
                    f.write(f"测量时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"图像文件: {os.path.basename(self.image_path)}\n")
                    f.write(f"镜片高度: {self.measurement_results['lens_height']:.2f} mm\n")
                    f.write("-"*40 + "\n")
                    f.write(f"左眼瞳高: {self.measurement_results['left_height']:.2f} mm\n")
                    f.write(f"右眼瞳高: {self.measurement_results['right_height']:.2f} mm\n")
                    f.write(f"平均瞳高: {self.measurement_results['avg_height']:.2f} mm\n")
                    f.write(f"瞳高差异: {self.measurement_results['diff']:.2f} mm\n")
                    f.write(f"状态评估: {self.measurement_results['status']}\n")
                    f.write("="*40 + "\n")
                self.status_bar.showMessage(f"测量报告已保存至: {save_path}")
            except Exception as e:
                QMessageBox.critical(self, "保存错误", f"保存报告文件时出错: {str(e)}")
            
def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    calculator = PupilHeightCalculator()
    calculator.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()