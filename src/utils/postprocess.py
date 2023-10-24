import numpy as np
import cv2


def preprocess(image, net_inshape, rgb_mean=(104, 117, 123)):
    im_shape = image.shape
    resize = float(net_inshape[0]) / float(im_shape[0])
    img = cv2.resize(image, net_inshape[::-1])
    img = np.float32(img)
    img -= rgb_mean
    img = img.transpose(2, 0, 1)

    return (img, resize)


def np_decode(loc, priors, variances=[0.1, 0.2]):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """
    priors = np.broadcast_to(priors, (loc.shape[0], loc.shape[1], 4))
    boxes = np.concatenate(
        (
            priors[..., :2] + loc[..., :2] * variances[0] * priors[..., 2:],
            priors[..., 2:] * np.exp(loc[..., 2:] * variances[1]),
        ),
        axis=2,
    )

    boxes[..., :2] -= boxes[..., 2:] / 2
    boxes[..., 2:] += boxes[..., :2]

    return boxes


def np_decode_landm(pre, priors, variances=[0.1, 0.2]):
    """Decode landm from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        pre (tensor): landm predictions for loc layers,
            Shape: [num_priors,10]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded landm predictions
    """
    priors = np.broadcast_to(priors, (pre.shape[0], pre.shape[1], 4))
    landms = np.concatenate(
        (
            priors[..., :2] + pre[..., :2] * variances[0] * priors[..., 2:],
            priors[..., :2] + pre[..., 2:4] * variances[0] * priors[..., 2:],
            priors[..., :2] + pre[..., 4:6] * variances[0] * priors[..., 2:],
            priors[..., :2] + pre[..., 6:8] * variances[0] * priors[..., 2:],
            priors[..., :2] + pre[..., 8:10] * variances[0] * priors[..., 2:],
        ),
        axis=2,
    )

    return landms


class DecodePostProcess:
    def __init__(self, net_inshape):
        self.scale_box = np.array([net_inshape[1], net_inshape[0]] * 2)
        self.scale_lamk = np.array([net_inshape[1], net_inshape[0]] * 5)

    def __call__(self, scores, boxes, landms, resize, detect_thresh=0.95):
        scores = scores[:, 1]
        boxes = boxes * self.scale_box / resize
        landms = landms * self.scale_lamk / resize

        # Ignore low scores
        inds = np.where(scores > 0.2)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # Keep top-K before NMS
        order = np.argsort(scores)[::-1][:5000]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # Do NMS
        keep = self.nms(boxes, scores, 0.4)
        dets = np.hstack((boxes, scores[:, np.newaxis]))
        dets = dets[keep]
        landms = landms[keep]

        # Keep top-K faster NMS
        dets = dets[:150]
        landms = landms[:150]

        dets = np.hstack((dets, landms))
        dets = dets[dets[:, 4] > detect_thresh]
        dets = dets[np.argsort(dets[:, 0])]

        return dets

    @staticmethod
    def nms(boxes, scores, overlap_thresh):
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            intersection = w * h

            overlap = intersection / (areas[i] + areas[order[1:]] - intersection)

            inds = np.where(overlap <= overlap_thresh)[0]
            order = order[inds + 1]

        return keep
