import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import optuna

# Variable para controlar la visualización de las imágenes
show_images = False

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    return blurred

def detect_edges(image, threshold1, threshold2):
    edges = cv2.Canny(image, threshold1, threshold2)
    return edges

def find_circles(image, dp, min_dist, param1, param2, min_radius, max_radius):
    circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, dp=dp, minDist=min_dist, param1=param1, param2=param2, minRadius=min_radius, maxRadius=max_radius)
    return circles

def draw_circles(image, circles):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    if circles is not None and len(circles) > 0:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            if r < 50:  # Ignorar círculos demasiado grandes
                cv2.circle(mask, (x, y), r, 255, -1)
    return mask.astype(np.uint8)

def detect_color_changes(image, lower_color, upper_color):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_image, lower_color, upper_color)
    edges = cv2.Canny(mask, 50, 150)
    return edges

def calculate_fractal_dimension(contour):
    x, y, w, h = cv2.boundingRect(contour)
    roi = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(roi, [contour], -1, 1, thickness=cv2.FILLED)
    max_size = min(w, h)
    min_size = 2
    num_boxes = []
    box_sizes = []
    size = max_size
    while size >= min_size:
        num_box = 0
        for i in range(0, w, size):
            for j in range(0, h, size):
                if np.any(roi[j:j+size, i:i]):
                    num_box += 1
        if num_box > 0:
            num_boxes.append(num_box)
            box_sizes.append(size)
        size //= 2
    num_boxes = np.array(num_boxes)
    box_sizes = np.array(box_sizes)
    if len(box_sizes) > 1 and len(num_boxes) > 1:
        coeffs = np.polyfit(np.log(box_sizes), np.log(num_boxes), 1)
        return -coeffs[0]
    else:
        return 0

def analyze_contours(image, contours, min_fractal_dim=1.2):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for contour in contours:
        fractal_dim = calculate_fractal_dimension(contour)
        if fractal_dim < min_fractal_dim:
            cv2.drawContours(mask, [contour], 0, 255, thickness=cv2.FILLED)
    edges = cv2.Canny(mask, 50, 150)
    return edges

def combined_detection(image, edges, circles, color_changes, fractal, weights, threshold):
    combined = weights[0] * circles + weights[1] * edges + weights[2] * color_changes + weights[3] * fractal
    combined[combined < threshold] = 0
    combined[combined >= threshold] = 255
    combined = combined.astype(np.uint8)
    return combined

def evaluate_combination(image, combined_mask):
    result_image = image.copy()
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) < 2500:  # Ignorar contornos demasiado grandes
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            cv2.circle(result_image, center, radius, (0, 255, 0), 2)
    return result_image

def read_yolo_labels(label_path, img_shape):
    labels = []
    with open(label_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            x_center, y_center, width, height = [float(p) for p in parts[1:]]
            x_center *= img_shape[1]
            y_center *= img_shape[0]
            width *= img_shape[1]
            height *= img_shape[0]
            labels.append((x_center, y_center, width, height))
    return labels

def iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xi1 = max(x1 - w1 / 2, x2 - w2 / 2)
    yi1 = max(y1 - h1 / 2, y2 - h2 / 2)
    xi2 = min(x1 + w1 / 2, x2 + w2 / 2)
    yi2 = min(y1 + h1 / 2, y2 + h2 / 2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area

def evaluate_model(predictions, labels, iou_threshold=0.5):
    tp = 0
    fp = 0
    fn = 0
    for pred in predictions:
        if any(iou(pred, label) > iou_threshold for label in labels):
            tp += 1
        else:
            fp += 1
    for label in labels:
        if not any(iou(pred, label) > iou_threshold for pred in predictions):
            fn += 1
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    return precision, recall, tp, fp, fn

# Cargar datos
base_dir = 'Aerial Airport.v1-v1.yolov9'
subsets = ['train', 'test', 'valid']

def get_image_label_paths(subset):
    image_paths = [os.path.join(base_dir, subset, 'images', f) for f in os.listdir(os.path.join(base_dir, subset, 'images'))]
    label_paths = [os.path.join(base_dir, subset, 'labels', f.replace('.jpg', '.txt')) for f in os.listdir(os.path.join(base_dir, subset, 'images'))]
    return image_paths, label_paths

train_images, train_labels = get_image_label_paths('train')
val_images, val_labels = get_image_label_paths('valid')
test_images, test_labels = get_image_label_paths('test')

# Función objetivo para Optuna
def objective(trial):
    dp = trial.suggest_float('dp', 1.0, 2.0)
    min_dist = trial.suggest_int('min_dist', 10, 50)
    param1 = trial.suggest_int('param1', 20, 100)
    param2 = trial.suggest_int('param2', 20, 100)
    min_radius = trial.suggest_int('min_radius', 5, 30)
    max_radius = trial.suggest_int('max_radius', 30, 50)
    threshold1 = trial.suggest_int('threshold1', 30, 100)
    threshold2 = trial.suggest_int('threshold2', 100, 200)
    weight_circles = trial.suggest_float('weight_circles', 0.0, 1.0)
    weight_edges = trial.suggest_float('weight_edges', 0.0, 1.0)
    weight_color_changes = trial.suggest_float('weight_color_changes', 0.0, 1.0)
    weight_fractal = trial.suggest_float('weight_fractal', 0.0, 1.0)
    threshold = trial.suggest_int('threshold', 50, 200)
    lower_color = np.array([0, 0, trial.suggest_int('lower_color_v', 100, 255)])
    upper_color = np.array([180, 30, trial.suggest_int('upper_color_v', 100, 255)])
    
    weights = (weight_circles, weight_edges, weight_color_changes, weight_fractal)

    precision_scores = []
    recall_scores = []
    for img_path, lbl_path in zip(val_images, val_labels):
        image = cv2.imread(img_path)
        preprocessed_image = preprocess_image(image)
        circles = find_circles(preprocessed_image, dp, min_dist, param1, param2, min_radius, max_radius)
        edges = detect_edges(preprocessed_image, threshold1, threshold2)
        color_changes = detect_color_changes(image, lower_color, upper_color)
        color_change_contours, _ = cv2.findContours(color_changes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        fractal_mask = analyze_contours(image, color_change_contours)
        combined_circle_mask = draw_circles(image, circles)
        combined_mask = combined_detection(image, edges, combined_circle_mask, color_changes, fractal_mask, weights, threshold)
        
        combined_contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        predictions = [cv2.boundingRect(contour) for contour in combined_contours]
        labels = read_yolo_labels(lbl_path, image.shape)
        
        precision, recall, tp, fp, fn = evaluate_model(predictions, labels)
        precision_scores.append(precision)
        recall_scores.append(recall)
    
    avg_precision = np.mean(precision_scores)
    avg_recall = np.mean(recall_scores)
    
    score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if avg_precision + avg_recall > 0 else 0
    
    return score

# Crear un estudio de Optuna y optimizar
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

# Obtener los mejores parámetros
best_params = study.best_params
print("Mejores parámetros:", best_params)

# Mostrar resultados con los mejores parámetros
for img_path, lbl_path in zip(val_images, val_labels):
    image = cv2.imread(img_path)
    preprocessed_image = preprocess_image(image)
    circles = find_circles(preprocessed_image, best_params['dp'], best_params['min_dist'], best_params['param1'], best_params['param2'], best_params['min_radius'], best_params['max_radius'])
    edges = detect_edges(preprocessed_image, best_params['threshold1'], best_params['threshold2'])
    color_changes = detect_color_changes(image, np.array([0, 0, best_params['lower_color_v']]), np.array([180, 30, best_params['upper_color_v']]))
    color_change_contours, _ = cv2.findContours(color_changes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fractal_mask = analyze_contours(image, color_change_contours)
    combined_circle_mask = draw_circles(image, circles)
    combined_mask = combined_detection(image, edges, combined_circle_mask, color_changes, fractal_mask, (best_params['weight_circles'], best_params['weight_edges'], best_params['weight_color_changes'], best_params['weight_fractal']), best_params['threshold'])
    detected_image = evaluate_combination(image, combined_mask)

    labels = read_yolo_labels(lbl_path, image.shape)

    real_image = image.copy()
    for label in labels:
        x_center, y_center, width, height = label
        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)
        cv2.rectangle(real_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if not show_images:
        plt.figure(figsize=(18, 12))
        plt.subplot(231), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Original Image')
        plt.subplot(232), plt.imshow(edges, cmap='gray'), plt.title('Edges')
        plt.subplot(233), plt.imshow(combined_circle_mask, cmap='gray'), plt.title('Circle Mask')
        plt.subplot(234), plt.imshow(color_changes, cmap='gray'), plt.title('Color Change Mask')
        plt.subplot(235), plt.imshow(fractal_mask, cmap='gray'), plt.title('Fractal Mask')
        plt.subplot(236), plt.imshow(cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB)), plt.title('Detected Image')
        plt.show()
    
        plt.figure(figsize=(12, 6))
        plt.subplot(121), plt.imshow(cv2.cvtColor(real_image, cv2.COLOR_BGR2RGB)), plt.title('Real Labels')
        plt.subplot(122), plt.imshow(cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB)), plt.title('Detected Objects')
        plt.show()
