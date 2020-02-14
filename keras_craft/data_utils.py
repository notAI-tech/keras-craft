import cv2
import numpy as np
from scipy import spatial
from shapely import geometry

# taken from https://stackoverflow.com/a/44659589
def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    r = 1

    if not (width or height):
        return image, r

    if height:
        if h <= height:
            return image, r

        r = height / float(h)
        dim = (int(w * r), height)

    else:
        if w <= width:
            return image, r

        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation = inter)

    return resized, r

    
def _read_image(image, max_width, max_height):
    if isinstance(image, type('')):
        image = cv2.imread(image)
        
    image, scale = image_resize(image, width=max_width, height=max_height)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = image.astype('float32')
    mean = np.array([0.485, 0.456, 0.406])
    variance = np.array([0.229, 0.224, 0.225])

    image -= mean * 255
    image /= variance * 255
    return image, scale

def _pad_image_bottom_right(cv2_image, bottom, right, color=[0, 0, 0]):
    return cv2.copyMakeBorder(cv2_image, 0, bottom, 0, right, cv2.BORDER_CONSTANT, value = color)

def read_images(images, max_width, max_height=None):
    images = [_read_image(image, max_width, max_height) for image in images]
    scales = [i[1] for i in images]
    images = [i[0] for i in images]
    
    shapes = [image.shape for image in images]

    if len(set(shapes)) > 1:
        max_height = max(shapes, key=lambda x: x[0])[0]
        max_width = max(shapes, key=lambda x: x[1])[1]

        images = [_pad_image_bottom_right(image, bottom=max_height - shapes[i][0], right=max_width - shapes[i][1]) for i, image in enumerate(images)]

    images = np.asarray(images)

    return images, scales, shapes


def keras_pred_to_boxes(y_pred, detection_threshold=0.7, text_threshold=0.4, link_threshold=0.4, size_threshold=10, shapes=None, scales=None):
    box_groups = []
    for pred_i, y_pred_cur in enumerate(y_pred):
        # Prepare data
        textmap = y_pred_cur[..., 0].copy()
        linkmap = y_pred_cur[..., 1].copy()

        scale = 1
        if scales:
            scale = scales[pred_i]

        if shapes:
            img_h, img_w = shapes[pred_i][:2]
        else:
            img_h, img_w = textmap.shape

        _, text_score = cv2.threshold(textmap,
                                      thresh=text_threshold,
                                      maxval=1,
                                      type=cv2.THRESH_BINARY)
        _, link_score = cv2.threshold(linkmap,
                                      thresh=link_threshold,
                                      maxval=1,
                                      type=cv2.THRESH_BINARY)
        n_components, labels, stats, _ = cv2.connectedComponentsWithStats(np.clip(
            text_score + link_score, 0, 1).astype('uint8'),
                                                                          connectivity=4)
        boxes = []
        for component_id in range(1, n_components):
            # Filter by size
            size = stats[component_id, cv2.CC_STAT_AREA]

            if size < size_threshold:
                continue

            # If the maximum value within this connected component is less than
            # text threshold, we skip it.
            if np.max(textmap[labels == component_id]) < detection_threshold:
                continue

            # Make segmentation map. It is 255 where we find text, 0 otherwise.
            segmap = np.zeros_like(textmap)
            segmap[labels == component_id] = 255
            segmap[np.logical_and(link_score, text_score)] = 0
            x, y, w, h = [
                stats[component_id, key] for key in
                [cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT]
            ]

            # Expand the elements of the segmentation map
            niter = int(np.sqrt(size * min(w, h) / (w * h)) * 2)
            sx, sy = max(x - niter, 0), max(y - niter, 0)
            ex, ey = min(x + w + niter + 1, img_w), min(y + h + niter + 1, img_h)
            segmap[sy:ey, sx:ex] = cv2.dilate(
                segmap[sy:ey, sx:ex],
                cv2.getStructuringElement(cv2.MORPH_RECT, (1 + niter, 1 + niter)))

            # Make rotated box from contour
            contours = cv2.findContours(segmap.astype('uint8'),
                                        mode=cv2.RETR_TREE,
                                        method=cv2.CHAIN_APPROX_SIMPLE)[-2]
            contour = contours[0]
            box = cv2.boxPoints(cv2.minAreaRect(contour))

            # Check to see if we have a diamond
            w, h = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
            box_ratio = max(w, h) / (min(w, h) + 1e-5)

            if abs(1 - box_ratio) <= 0.1:
                l, r = contour[:, 0, 0].min(), contour[:, 0, 0].max()
                t, b = contour[:, 0, 1].min(), contour[:, 0, 1].max()
                box = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)
            else:
                # Make clock-wise order
                box = np.array(np.roll(box, 4 - box.sum(axis=1).argmin(), 0))
                
            boxes.append(2 * box  * (1/scale))

        box_groups.append(boxes)

    return box_groups


def draw_boxes_on_image(image_path, boxes, out_path=None, color=(255, 0, 0), thickness=5):
    if not out_path:
        image_type = image_path.split('.')[-1]
        out_path = '.'.join(image_path.split('.')[:-1]) + '.craft.' + image_type

    image = cv2.imread(image_path)

    for box in boxes:
        cv2.polylines(img=image, pts=box[np.newaxis].astype('int32'), color=color, thickness=thickness, isClosed=True)

    cv2.imwrite(out_path, image)

    return out_path

def _get_rotated_box(points):
    mp = geometry.MultiPoint(points=points)
    pts = np.array(list(zip(*mp.minimum_rotated_rectangle.exterior.xy)))[:-1]  # noqa: E501

    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]

    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost

    D = spatial.distance.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]

    pts = np.array([tl, tr, br, bl], dtype="float32")

    rotation = np.arctan((tl[0] - bl[0]) / (tl[1] - bl[1]))
    return pts, rotation

def get_crop_in_box(image, box, target_height=None, target_width=None, margin=5):
    cval = (0, 0, 0) if len(image.shape) == 3 else 0

    box, _ = _get_rotated_box(box)
    _, _, w, h = cv2.boundingRect(box)

    if target_width is None and target_height is None:
        target_width = w
        target_height = h
    
    scale = min(target_width / w, target_height / h)
    M = cv2.getPerspectiveTransform(src=box,
                                    dst=np.array([[margin, margin], [scale * w - margin, margin],
                                                  [scale * w - margin, scale * h - margin],
                                                  [margin, scale * h - margin]]).astype('float32'))
    crop = cv2.warpPerspective(image, M, dsize=(int(scale * w), int(scale * h)))
    target_shape = (target_height, target_width, 3) if len(image.shape) == 3 else (target_height,
                                                                                   target_width)
    full = (np.zeros(target_shape) + cval).astype('uint8')
    full[:crop.shape[0], :crop.shape[1]] = crop

    return full