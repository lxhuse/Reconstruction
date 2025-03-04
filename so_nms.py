import numpy as np

def so_nms(boxes, scores, score_threshold=0.5):
    """
    SO-NMS

    parameters:
        boxes (np.ndarray): shape (N,4), format[x1,y1,x2,y2]
        scores (np.ndarray): shape (N,)
        score_threshold (float): default 0.5

    Return:
        (np.ndarray, np.ndarray): box , score
    """
    # Filtering low confidence detection frames
    mask = scores >= score_threshold
    selected_boxes = boxes[mask]
    selected_scores = scores[mask]

    # Returns an empty result if there are no test boxes that satisfy the condition
    if len(selected_boxes) == 0:
        return np.zeros((0, 4)), np.zeros((0,))

    # Calculate the minimum envelope rectangle
    x1 = np.min(selected_boxes[:, 0])
    y1 = np.min(selected_boxes[:, 1])
    x2 = np.max(selected_boxes[:, 2])
    y2 = np.max(selected_boxes[:, 3])
    envelope_box = np.array([[x1, y1, x2, y2]])

    # Calculation of the average score
    avg_score = np.mean(selected_scores)
    
    return envelope_box, np.array([avg_score])

# example
if __name__ == "__main__":
    # Example input data (format: N [x1,y1,x2,y2])
    boxes = np.array([
        [10, 10, 20, 20],
        [15, 15, 25, 25],
        [5, 5, 30, 30]
    ], dtype=np.float32)
    
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    
    # Conduct SO-NMS
    final_box, final_score = so_nms(boxes, scores, score_threshold=0.5)
    
    print("Final Bounding Box:", final_box)
    print("Final Score:", final_score)