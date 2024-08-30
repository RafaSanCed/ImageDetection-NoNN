import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import optuna

# Variable para controlar la visualización de las imágenes
show_images = False

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1)
    return blurred

def detect_edges(image, threshold1, threshold2):
    edges = cv2.Canny(image, threshold1, threshold2)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    return edges

def detect_color_changes_along_edges(image, edges, r_thresh, g_thresh, b_thresh):
    b_channel, g_channel, r_channel = cv2.split(image)
    color_change_mask = np.zeros_like(edges)
    
    for y in range(edges.shape[0]):
        for x in range(edges.shape[1]):
            if edges[y, x] > 0:
                b_pixel = b_channel[y, x]
                g_pixel = g_channel[y, x]
                r_pixel = r_channel[y, x]
                if (b_pixel > b_thresh and g_pixel > g_thresh and r_pixel > r_thresh):
                    color_change_mask[y, x] = 255
    
    return color_change_mask

def combined_detection(edges, color_changes, weights, threshold):
    combined = weights[0] * edges + weights[1] * color_changes
    combined[combined < threshold] = 0
    combined[combined >= threshold] = 255
    combined = combined.astype(np.uint8)
    return combined

def evaluate_combination(image, combined_mask):
    result_image = image.copy()
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Ignorar contornos demasiado pequeños
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
            _, x_center, y_center, width, height = [float(p) for p in parts]
            x_center *= img_shape[1]
            y_center *= img_shape[0]
            width *= img_shape[1]
            height *= img_shape[0]
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2
            labels.append((x1, y1, x2, y2))
    return np.array(labels)

def is_inside(pred_box, label_box):
    px1, py1, px2, py2 = pred_box
    lx1, ly1, lx2, ly2 = label_box

    return lx1 <= px1 <= lx2 and ly1 <= py1 <= ly2 and lx1 <= px2 <= lx2 and ly1 <= py2 <= ly2

def evaluate_model(predictions, labels):
    tp = 0
    fp = 0
    fn = 0
    tp_boxes = []
    fp_boxes = []
    fn_boxes = []

    detected_labels = np.zeros(len(labels), dtype=bool)

    for pred in predictions:
        matched = False
        for idx, label in enumerate(labels):
            if is_inside(pred, label):
                tp += 1
                tp_boxes.append(label)  # Use the real label box
                detected_labels[idx] = True
                matched = True
                break
        if not matched:
            fp += 1
            fp_boxes.append(pred)

    for idx, label in enumerate(labels):
        if not detected_labels[idx]:
            fn += 1
            fn_boxes.append(label)

    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    return precision, recall, tp, fp, fn, tp_boxes, fp_boxes, fn_boxes

# Cargar datos
base_dir = 'Aerial Airport.v1-v1.yolov9'
subset = 'valid'

def get_image_label_paths(subset):
    image_paths = [os.path.join(base_dir, subset, 'images', f) for f in os.listdir(os.path.join(base_dir, subset, 'images'))]
    label_paths = [os.path.join(base_dir, subset, 'labels', f.replace('.jpg', '.txt')) for f in os.listdir(os.path.join(base_dir, subset, 'images'))]
    return image_paths, label_paths

val_images, val_labels = get_image_label_paths(subset)

# Seleccionar una sola imagen para la optimización
img_path = val_images[0]
lbl_path = val_labels[0]

image = cv2.imread(img_path)
labels = read_yolo_labels(lbl_path, image.shape)

# Función objetivo para Optuna
def objective(trial):
    threshold1 = trial.suggest_int('threshold1', 20, 50)
    threshold2 = trial.suggest_int('threshold2', 50, 100)
    weight_edges = trial.suggest_float('weight_edges', 0.0, 1.0)
    weight_color_changes = trial.suggest_float('weight_color_changes', 0.0, 1.0)
    threshold = trial.suggest_int('threshold', 30, 100)
    r_thresh = trial.suggest_int('r_thresh', 180, 255)
    g_thresh = trial.suggest_int('g_thresh', 180, 255)
    b_thresh = trial.suggest_int('b_thresh', 180, 255)
    
    weights = (weight_edges, weight_color_changes)

    preprocessed_image = preprocess_image(image)
    edges = detect_edges(preprocessed_image, threshold1, threshold2)
    color_changes = detect_color_changes_along_edges(image, edges, r_thresh, g_thresh, b_thresh)
    combined_mask = combined_detection(edges, color_changes, weights, threshold)
    
    combined_contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    predictions = [cv2.boundingRect(contour) for contour in combined_contours]
    predictions = [(x, y, x + w, y + h) for x, y, w, h in predictions]

    # Eliminar predicciones que superen los 50 píxeles
    predictions = [box for box in predictions if (box[2] - box[0]) <= 50 and (box[3] - box[1]) <= 50]
    
    precision, recall, tp, fp, fn, tp_boxes, fp_boxes, fn_boxes = evaluate_model(predictions, labels)
    
    # Mostrar resultados para la imagen en cada iteración
    if val_images:
        detected_image = evaluate_combination(image, combined_mask)

        real_image = image.copy()
        detected_objects_image = image.copy()
        for label in labels:
            x1, y1, x2, y2 = label
            cv2.rectangle(real_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        
        for box in predictions:
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            if w < 20:
                x2 = x1 + 20
            if h < 20:
                y2 = y1 + 20
            cv2.rectangle(detected_objects_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 2)  # Predictions (yellow)

        result_image = image.copy()
        for box in predictions:
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            if w < 10:
                x2 = x1 + 10
            if h < 10:
                y2 = y1 + 10
            cv2.rectangle(result_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 2)  # Predictions (yellow)
        for box in tp_boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(result_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # True Positives (green)
        for box in fp_boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(result_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)  # False Positives (red)
        for box in fn_boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(result_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)  # False Negatives (blue)

        if not show_images:
            plt.figure(figsize=(18, 12))
            plt.subplot(231), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Original Image')
            plt.subplot(232), plt.imshow(edges, cmap='gray'), plt.title('Edges')
            plt.subplot(233), plt.imshow(color_changes, cmap='gray'), plt.title('Color Change Mask')
            plt.subplot(234), plt.imshow(cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB)), plt.title('Detected Image')

            # Mostrar etiquetas detectadas en la primera figura
            for box in predictions:
                x1, y1, x2, y2 = box
                w, h = x2 - x1, y2 - y1
                if w < 20:
                    x2 = x1 + 20
                if h < 20:
                    y2 = y1 + 20
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 2)  # Predictions (yellow)

            plt.subplot(235), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Predictions (yellow)')
            plt.show()

            plt.figure(figsize=(12, 6))
            plt.subplot(121), plt.imshow(cv2.cvtColor(real_image, cv2.COLOR_BGR2RGB)), plt.title('Real Labels')
            plt.subplot(122), plt.imshow(cv2.cvtColor(detected_objects_image, cv2.COLOR_BGR2RGB)), plt.title('Detected Objects (Predictions in Yellow)')
            plt.show()

            plt.figure(figsize=(12, 6))
            plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)), plt.title('TP (green), FP (red), FN (blue), Predictions (yellow)')
            plt.show()

    print(f"Trial {trial.number}: Precision: {precision:.4f}, Recall: {recall:.4f}, TP: {tp}, FP: {fp}, FN: {fn}")
    return precision

# Crear un estudio de Optuna y optimizar
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

# Obtener los mejores parámetros
best_params = study.best_params
print("Mejores parámetros:", best_params)

# Mostrar resultados con los mejores parámetros
threshold1 = best_params['threshold1']
threshold2 = best_params['threshold2']
weight_edges = best_params['weight_edges']
weight_color_changes = best_params['weight_color_changes']
threshold = best_params['threshold']
r_thresh = best_params['r_thresh']
g_thresh = best_params['g_thresh']
b_thresh = best_params['b_thresh']

weights = (weight_edges, weight_color_changes)

preprocessed_image = preprocess_image(image)
edges = detect_edges(preprocessed_image, threshold1, threshold2)
color_changes = detect_color_changes_along_edges(image, edges, r_thresh, g_thresh, b_thresh)
combined_mask = combined_detection(edges, color_changes, weights, threshold)

combined_contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
predictions = [cv2.boundingRect(contour) for contour in combined_contours]
predictions = [(x, y, x + w, y + h) for x, y, w, h in predictions]

# Eliminar predicciones que superen los 50 píxeles
predictions = [box for box in predictions if (box[2] - box[0]) <= 50 and (box[3] - box[1]) <= 50]

precision, recall, tp, fp, fn, tp_boxes, fp_boxes, fn_boxes = evaluate_model(predictions, labels)

# Mostrar resultados
detected_image = evaluate_combination(image, combined_mask)

real_image = image.copy()
detected_objects_image = image.copy()
for label in labels:
    x1, y1, x2, y2 = label
    cv2.rectangle(real_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

for box in predictions:
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    if w < 20:
        x2 = x1 + 20
    if h < 20:
        y2 = y1 + 20
    cv2.rectangle(detected_objects_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 2)  # Predictions (yellow)

result_image = image.copy()
for box in predictions:
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    if w < 10:
        x2 = x1 + 10
    if h < 10:
        y2 = y1 + 10
    cv2.rectangle(result_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 2)  # Predictions (yellow)
for box in tp_boxes:
    x1, y1, x2, y2 = box
    cv2.rectangle(result_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # True Positives (green)
for box in fp_boxes:
    x1, y1, x2, y2 = box
    cv2.rectangle(result_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)  # False Positives (red)
for box in fn_boxes:
    x1, y1, x2, y2 = box
    cv2.rectangle(result_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)  # False Negatives (blue)

if not show_images:
    plt.figure(figsize=(18, 12))
    plt.subplot(231), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Original Image')
    plt.subplot(232), plt.imshow(edges, cmap='gray'), plt.title('Edges')
    plt.subplot(233), plt.imshow(color_changes, cmap='gray'), plt.title('Color Change Mask')
    plt.subplot(234), plt.imshow(cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB)), plt.title('Detected Image')

    # Mostrar etiquetas detectadas en la primera figura
    for box in predictions:
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        if w < 20:
            x2 = x1 + 20
        if h < 20:
            y2 = y1 + 20
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 2)  # Predictions (yellow)

    plt.subplot(235), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Predictions (yellow)')
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.subplot(121), plt.imshow(cv2.cvtColor(real_image, cv2.COLOR_BGR2RGB)), plt.title('Real Labels')
    plt.subplot(122), plt.imshow(cv2.cvtColor(detected_objects_image, cv2.COLOR_BGR2RGB)), plt.title('Detected Objects (Predictions in Yellow)')
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)), plt.title('TP (green), FP (red), FN (blue), Predictions (yellow)')
    plt.show()

print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, TP: {tp}, FP: {fp}, FN: {fn}")
