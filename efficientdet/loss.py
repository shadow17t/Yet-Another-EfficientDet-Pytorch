import torch
import torch.nn as nn
import cv2
import numpy as np
import torch.nn.functional as F

from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import postprocess, invert_affine, display


def calc_iou(a, b):
    # a(anchor) [boxes, (y1, x1, y2, x2)]
    # b(gt, coco-style) [boxes, (x1, y1, x2, y2)]

    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) #bbox area

    iw = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 1])
    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)

    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih
    ua = torch.clamp(ua, min=1e-8)
    intersection = iw * ih
    IoU = intersection / ua

    return IoU

def calc_diou(a, b):
    # Menghitung IoU
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 1])
    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)

    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih
    ua = torch.clamp(ua, min=1e-8)
    intersection = iw * ih
    iou = intersection / ua

    # Menghitung jarak antar pusat
    a_center_x = torch.unsqueeze(a[:, 3] + a[:, 1], dim=1) / 2
    a_center_y = torch.unsqueeze(a[:, 2] + a[:, 0], dim=1) / 2
    b_center_x = (b[:, 2] + b[:, 0]) / 2
    b_center_y = (b[:, 3] + b[:, 1]) / 2

    center_dist = (a_center_x - b_center_x)**2 + (a_center_y - b_center_y)**2

    # Menghitung diagonal terpanjang dari bounding box terluar
    enclose_x1 = torch.min(torch.unsqueeze(a[:, 1], dim=1), b[:, 0])
    enclose_y1 = torch.min(torch.unsqueeze(a[:, 0], dim=1), b[:, 1])
    enclose_x2 = torch.max(torch.unsqueeze(a[:, 3], dim=1), b[:, 2])
    enclose_y2 = torch.max(torch.unsqueeze(a[:, 2], dim=1), b[:, 3])
    
    enclose_diagonal = (enclose_x2 - enclose_x1)**2 + (enclose_y2 - enclose_y1)**2
    enclose_diagonal = torch.clamp(enclose_diagonal, min=1e-8)

    # Menghitung DIoU
    diou = iou - center_dist / enclose_diagonal

    return diou

class FocalLoss(nn.Module):
    def __init__(self):
        super(FocalLoss, self).__init__()

    def forward(self, classifications, regressions, anchors, annotations, **kwargs):
        alpha = 0.25
        gamma = 2.0
        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []

        anchor = anchors[0, :, :]  # assuming all image sizes are the same, which it is
        dtype = anchors.dtype

        anchor_widths = anchor[:, 3] - anchor[:, 1]
        anchor_heights = anchor[:, 2] - anchor[:, 0]
        anchor_ctr_x = anchor[:, 1] + 0.5 * anchor_widths
        anchor_ctr_y = anchor[:, 0] + 0.5 * anchor_heights

        for j in range(batch_size):

            classification = classifications[j, :, :]
            regression = regressions[j, :, :]

            bbox_annotation = annotations[j]
            bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]

            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)
            
            if bbox_annotation.shape[0] == 0:
                if torch.cuda.is_available():
                    
                    alpha_factor = torch.ones_like(classification) * alpha
                    alpha_factor = alpha_factor.cuda()
                    alpha_factor = 1. - alpha_factor
                    focal_weight = classification
                    focal_weight = alpha_factor * torch.pow(focal_weight, gamma)
                    
                    bce = -(torch.log(1.0 - classification))
                    
                    cls_loss = focal_weight * bce
                    
                    regression_losses.append(torch.tensor(0).to(dtype).cuda())
                    classification_losses.append(cls_loss.sum())
                else:
                    
                    alpha_factor = torch.ones_like(classification) * alpha
                    alpha_factor = 1. - alpha_factor
                    focal_weight = classification
                    focal_weight = alpha_factor * torch.pow(focal_weight, gamma)
                    
                    bce = -(torch.log(1.0 - classification))
                    
                    cls_loss = focal_weight * bce
                    
                    regression_losses.append(torch.tensor(0).to(dtype))
                    classification_losses.append(cls_loss.sum())

                continue
                
            IoU = calc_iou(anchor[:, :], bbox_annotation[:, :4])

            IoU_max, IoU_argmax = torch.max(IoU, dim=1)

            # compute the loss for classification
            targets = torch.ones_like(classification) * -1
            if torch.cuda.is_available():
                targets = targets.cuda()

            targets[torch.lt(IoU_max, 0.4), :] = 0

            positive_indices = torch.ge(IoU_max, 0.5)

            num_positive_anchors = positive_indices.sum()

            assigned_annotations = bbox_annotation[IoU_argmax, :]

            targets[positive_indices, :] = 0
            targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1

            alpha_factor = torch.ones_like(targets) * alpha
            if torch.cuda.is_available():
                alpha_factor = alpha_factor.cuda()

            alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification)
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

            bce = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))

            cls_loss = focal_weight * bce

            zeros = torch.zeros_like(cls_loss)
            if torch.cuda.is_available():
                zeros = zeros.cuda()
            cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, zeros)

            classification_losses.append(cls_loss.sum() / torch.clamp(num_positive_anchors.to(dtype), min=1.0))

            if positive_indices.sum() > 0:
                assigned_annotations = assigned_annotations[positive_indices, :]

                anchor_widths_pi = anchor_widths[positive_indices]
                anchor_heights_pi = anchor_heights[positive_indices]
                anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
                anchor_ctr_y_pi = anchor_ctr_y[positive_indices]

                gt_widths = assigned_annotations[:, 2] - assigned_annotations[:, 0]
                gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
                gt_ctr_x = assigned_annotations[:, 0] + 0.5 * gt_widths
                gt_ctr_y = assigned_annotations[:, 1] + 0.5 * gt_heights

                # efficientdet style
                gt_widths = torch.clamp(gt_widths, min=1)
                gt_heights = torch.clamp(gt_heights, min=1)

                targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
                targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
                targets_dw = torch.log(gt_widths / anchor_widths_pi)
                targets_dh = torch.log(gt_heights / anchor_heights_pi)

                targets = torch.stack((targets_dy, targets_dx, targets_dh, targets_dw))
                targets = targets.t()

                regression_diff = torch.abs(targets - regression[positive_indices, :])

                regression_loss = torch.where(
                    torch.le(regression_diff, 1.0 / 9.0),
                    0.5 * 9.0 * torch.pow(regression_diff, 2),
                    regression_diff - 0.5 / 9.0
                )
                regression_losses.append(regression_loss.mean())
            else:
                if torch.cuda.is_available():
                    regression_losses.append(torch.tensor(0).to(dtype).cuda())
                else:
                    regression_losses.append(torch.tensor(0).to(dtype))

        # debug
        imgs = kwargs.get('imgs', None)
        if imgs is not None:
            regressBoxes = BBoxTransform()
            clipBoxes = ClipBoxes()
            obj_list = kwargs.get('obj_list', None)
            out = postprocess(imgs.detach(),
                              torch.stack([anchors[0]] * imgs.shape[0], 0).detach(), regressions.detach(), classifications.detach(),
                              regressBoxes, clipBoxes,
                              0.5, 0.3)
            imgs = imgs.permute(0, 2, 3, 1).cpu().numpy()
            imgs = ((imgs * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255).astype(np.uint8)
            imgs = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in imgs]
            display(out, imgs, obj_list, imshow=False, imwrite=True)

        return torch.stack(classification_losses).mean(dim=0, keepdim=True), \
               torch.stack(regression_losses).mean(dim=0, keepdim=True) * 50  # https://github.com/google/automl/blob/6fdd1de778408625c1faf368a327fe36ecd41bf7/efficientdet/hparams_config.py#L233

class FocalLossDIoU(nn.Module):
    def __init__(self):
        super(FocalLossDIoU, self).__init__()

    def forward(self, classifications, regressions, anchors, annotations, **kwargs):
        alpha = 0.25
        gamma = 2.0
        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []

        anchor = anchors[0, :, :]  # assuming all image sizes are the same, which it is
        dtype = anchors.dtype

        anchor_widths = anchor[:, 3] - anchor[:, 1]
        anchor_heights = anchor[:, 2] - anchor[:, 0]
        anchor_ctr_x = anchor[:, 1] + 0.5 * anchor_widths
        anchor_ctr_y = anchor[:, 0] + 0.5 * anchor_heights

        for j in range(batch_size):

            classification = classifications[j, :, :]
            regression = regressions[j, :, :]

            bbox_annotation = annotations[j]
            bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]

            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)
            
            if bbox_annotation.shape[0] == 0:
                if torch.cuda.is_available():
                    
                    alpha_factor = torch.ones_like(classification) * alpha
                    alpha_factor = alpha_factor.cuda()
                    alpha_factor = 1. - alpha_factor
                    focal_weight = classification
                    focal_weight = alpha_factor * torch.pow(focal_weight, gamma)
                    
                    bce = -(torch.log(1.0 - classification))
                    
                    cls_loss = focal_weight * bce
                    
                    regression_losses.append(torch.tensor(0).to(dtype).cuda())
                    classification_losses.append(cls_loss.sum())
                else:
                    
                    alpha_factor = torch.ones_like(classification) * alpha
                    alpha_factor = 1. - alpha_factor
                    focal_weight = classification
                    focal_weight = alpha_factor * torch.pow(focal_weight, gamma)
                    
                    bce = -(torch.log(1.0 - classification))
                    
                    cls_loss = focal_weight * bce
                    
                    regression_losses.append(torch.tensor(0).to(dtype))
                    classification_losses.append(cls_loss.sum())

                continue
                
            IoU = calc_iou(anchor[:, :], bbox_annotation[:, :4])

            IoU_max, IoU_argmax = torch.max(IoU, dim=1)

            # compute the loss for classification
            targets = torch.ones_like(classification) * -1
            if torch.cuda.is_available():
                targets = targets.cuda()

            targets[torch.lt(IoU_max, 0.4), :] = 0

            positive_indices = torch.ge(IoU_max, 0.5)

            num_positive_anchors = positive_indices.sum()

            assigned_annotations = bbox_annotation[IoU_argmax, :]

            targets[positive_indices, :] = 0
            targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1

            alpha_factor = torch.ones_like(targets) * alpha
            if torch.cuda.is_available():
                alpha_factor = alpha_factor.cuda()

            alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification)
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

            bce = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))

            cls_loss = focal_weight * bce

            zeros = torch.zeros_like(cls_loss)
            if torch.cuda.is_available():
                zeros = zeros.cuda()
            cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, zeros)

            classification_losses.append(cls_loss.sum() / torch.clamp(num_positive_anchors.to(dtype), min=1.0))

            if positive_indices.sum() > 0:
                assigned_annotations = assigned_annotations[positive_indices, :]

                anchor_widths_pi = anchor_widths[positive_indices]
                anchor_heights_pi = anchor_heights[positive_indices]
                anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
                anchor_ctr_y_pi = anchor_ctr_y[positive_indices]

                gt_widths = assigned_annotations[:, 2] - assigned_annotations[:, 0]
                gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
                gt_ctr_x = assigned_annotations[:, 0] + 0.5 * gt_widths
                gt_ctr_y = assigned_annotations[:, 1] + 0.5 * gt_heights

                # efficientdet style
                gt_widths = torch.clamp(gt_widths, min=1)
                gt_heights = torch.clamp(gt_heights, min=1)

                targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
                targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
                targets_dw = torch.log(gt_widths / anchor_widths_pi)
                targets_dh = torch.log(gt_heights / anchor_heights_pi)

                targets = torch.stack((targets_dy, targets_dx, targets_dh, targets_dw))
                targets = targets.t()

                regression_diff = torch.abs(targets - regression[positive_indices, :])

                # regression_loss = torch.where(
                #     torch.le(regression_diff, 1.0 / 9.0),
                #     0.5 * 9.0 * torch.pow(regression_diff, 2),
                #     regression_diff - 0.5 / 9.0
                # )

                # Menggunakan DIoU untuk regression loss
                anchors_pos = torch.stack([anchor_ctr_y_pi - anchor_heights_pi/2,
                                        anchor_ctr_x_pi - anchor_widths_pi/2,
                                        anchor_ctr_y_pi + anchor_heights_pi/2,
                                        anchor_ctr_x_pi + anchor_widths_pi/2], dim=1)
                diou = calc_diou(anchors_pos, assigned_annotations[:, :4])
                
                regression_loss = 1 - diou + torch.where(
                    torch.le(regression_diff, 1.0 / 9.0),
                    0.5 * 9.0 * torch.pow(regression_diff, 2),
                    regression_diff - 0.5 / 9.0
                ).sum(dim=1)
                regression_losses.append(regression_loss.mean())
            else:
                if torch.cuda.is_available():
                    regression_losses.append(torch.tensor(0).to(dtype).cuda())
                else:
                    regression_losses.append(torch.tensor(0).to(dtype))

        # debug
        imgs = kwargs.get('imgs', None)
        if imgs is not None:
            regressBoxes = BBoxTransform()
            clipBoxes = ClipBoxes()
            obj_list = kwargs.get('obj_list', None)
            out = postprocess(imgs.detach(),
                              torch.stack([anchors[0]] * imgs.shape[0], 0).detach(), regressions.detach(), classifications.detach(),
                              regressBoxes, clipBoxes,
                              0.5, 0.3)
            imgs = imgs.permute(0, 2, 3, 1).cpu().numpy()
            imgs = ((imgs * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255).astype(np.uint8)
            imgs = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in imgs]
            display(out, imgs, obj_list, imshow=False, imwrite=True)

        return torch.stack(classification_losses).mean(dim=0, keepdim=True), \
               torch.stack(regression_losses).mean(dim=0, keepdim=True) * 50  # https://github.com/google/automl/blob/6fdd1de778408625c1faf368a327fe36ecd41bf7/efficientdet/hparams_config.py#L233

# class VarifocalLoss(nn.Module):
#     def __init__(self, use_sigmoid=True, alpha=0.75, gamma=2.0, iou_weighted=True):
#         super(VarifocalLoss, self).__init__()
#         self.use_sigmoid = use_sigmoid
#         self.alpha = alpha
#         self.gamma = gamma
#         self.iou_weighted = iou_weighted

#     def forward(self, classifications, regressions, anchors, annotations, **kwargs):
#         alpha = 0.75
#         gamma = 2.0
#         # use_sigmoid=True
#         # iou_weighted=True
#         batch_size = classifications.shape[0]
#         classification_losses = []
#         regression_losses = []

#         anchor = anchors[0, :, :]  # assuming all image sizes are the same, which it is
#         dtype = anchors.dtype

#         anchor_widths = anchor[:, 3] - anchor[:, 1]
#         anchor_heights = anchor[:, 2] - anchor[:, 0]
#         anchor_ctr_x = anchor[:, 1] + 0.5 * anchor_widths
#         anchor_ctr_y = anchor[:, 0] + 0.5 * anchor_heights

#         for j in range(batch_size):

#             classification = classifications[j, :, :]
#             regression = regressions[j, :, :]

#             bbox_annotation = annotations[j]
#             bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]

#             classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)
            
#             if bbox_annotation.shape[0] == 0:
#                 if torch.cuda.is_available():
                    
#                     alpha_factor = torch.ones_like(classification) * alpha
#                     alpha_factor = alpha_factor.cuda()
#                     alpha_factor = 1. - alpha_factor
#                     focal_weight = classification
#                     focal_weight = alpha_factor * torch.pow(focal_weight, gamma)
                    
#                     bce = -(torch.log(1.0 - classification))
                    
#                     cls_loss = focal_weight * bce
                    
#                     regression_losses.append(torch.tensor(0).to(dtype).cuda())
#                     classification_losses.append(cls_loss.sum())
#                 else:
                    
#                     alpha_factor = torch.ones_like(classification) * alpha
#                     alpha_factor = 1. - alpha_factor
#                     focal_weight = classification
#                     focal_weight = alpha_factor * torch.pow(focal_weight, gamma)
                    
#                     bce = -(torch.log(1.0 - classification))
                    
#                     cls_loss = focal_weight * bce
                    
#                     regression_losses.append(torch.tensor(0).to(dtype))
#                     classification_losses.append(cls_loss.sum())

#                 continue
                
#             IoU = calc_diou(anchor[:, :], bbox_annotation[:, :4])

#             IoU_max, IoU_argmax = torch.max(IoU, dim=1)

#             # compute the loss for classification
#             targets = torch.ones_like(classification) * -1
#             if torch.cuda.is_available():
#                 targets = targets.cuda()

#             targets[torch.lt(IoU_max, 0.4), :] = 0

#             positive_indices = torch.ge(IoU_max, 0.5)

#             num_positive_anchors = positive_indices.sum()

#             assigned_annotations = bbox_annotation[IoU_argmax, :]

#             targets[positive_indices, :] = 0
#             targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = IoU_max[positive_indices]

#             alpha_factor = torch.ones_like(targets) * alpha
#             if torch.cuda.is_available():
#                 alpha_factor = alpha_factor.cuda()

#             if self.use_sigmoid:
#                 pred_sigmoid = torch.sigmoid(classification)
#             else:
#                 pred_sigmoid = classification

#             target_onehot = F.one_hot(targets.to(dtype), num_classes=classification.shape[-1])

#             if self.iou_weighted:
#                 focal_weight = torch.pow(IoU_max.unsqueeze(-1).expand_as(target_onehot) - pred_sigmoid, self.gamma)
#             else:
#                 focal_weight = torch.pow(target_onehot - pred_sigmoid, self.gamma)

#             cls_loss = self.alpha * focal_weight * (
#                 target_onehot * F.logsigmoid(classification) +
#                 (1 - target_onehot) * F.logsigmoid(-classification)
#             )

#             zeros = torch.zeros_like(cls_loss)
#             if torch.cuda.is_available():
#                 zeros = zeros.cuda()
#             cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, zeros)

#             # Tambahkan perhitungan IoU loss
#             iou_loss = 1 - IoU_max[positive_indices]

#             # Gabungkan losses
#             total_loss = cls_loss.sum() / torch.clamp(num_positive_anchors.to(dtype), min=1.0)
#             if positive_indices.sum() > 0:
#                 total_loss += iou_loss.sum() / num_positive_anchors.to(dtype)

#             classification_losses.append(total_loss)
#             # classification_losses.append(cls_loss.sum() / torch.clamp(num_positive_anchors.to(dtype), min=1.0))

#             if positive_indices.sum() > 0:
#                 assigned_annotations = assigned_annotations[positive_indices, :]

#                 anchor_widths_pi = anchor_widths[positive_indices]
#                 anchor_heights_pi = anchor_heights[positive_indices]
#                 anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
#                 anchor_ctr_y_pi = anchor_ctr_y[positive_indices]

#                 gt_widths = assigned_annotations[:, 2] - assigned_annotations[:, 0]
#                 gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
#                 gt_ctr_x = assigned_annotations[:, 0] + 0.5 * gt_widths
#                 gt_ctr_y = assigned_annotations[:, 1] + 0.5 * gt_heights

#                 # efficientdet style
#                 gt_widths = torch.clamp(gt_widths, min=1)
#                 gt_heights = torch.clamp(gt_heights, min=1)

#                 # targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
#                 # targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
#                 # targets_dw = torch.log(gt_widths / anchor_widths_pi)
#                 # targets_dh = torch.log(gt_heights / anchor_heights_pi)

#                 # targets = torch.stack((targets_dy, targets_dx, targets_dh, targets_dw))
#                 # targets = targets.t()

#                 # regression_diff = torch.abs(targets - regression[positive_indices, :])

#                 # regression_loss = torch.where(
#                 #     torch.le(regression_diff, 1.0 / 9.0),
#                 #     0.5 * 9.0 * torch.pow(regression_diff, 2),
#                 #     regression_diff - 0.5 / 9.0
#                 # )
#                 # Konversi prediksi regresi menjadi kotak yang sebenarnya
#                 pred_ctr_x = anchor_ctr_x_pi + regression[positive_indices, 1] * anchor_widths_pi
#                 pred_ctr_y = anchor_ctr_y_pi + regression[positive_indices, 0] * anchor_heights_pi
#                 pred_w = anchor_widths_pi * torch.exp(regression[positive_indices, 2])
#                 pred_h = anchor_heights_pi * torch.exp(regression[positive_indices, 3])

#                 pred_boxes = torch.stack([
#                     pred_ctr_x - pred_w / 2,
#                     pred_ctr_y - pred_h / 2,
#                     pred_ctr_x + pred_w / 2,
#                     pred_ctr_y + pred_h / 2
#                 ], dim=1)

#                 target_boxes = torch.stack([
#                     gt_ctr_x - gt_widths / 2,
#                     gt_ctr_y - gt_heights / 2,
#                     gt_ctr_x + gt_widths / 2,
#                     gt_ctr_y + gt_heights / 2
#                 ], dim=1)
#                 # Hitung DIoU loss
#                 diou = calc_diou(pred_boxes, target_boxes)
#                 regression_loss = 1 - diou
#                 regression_losses.append(regression_loss.mean())
#             else:
#                 if torch.cuda.is_available():
#                     regression_losses.append(torch.tensor(0).to(dtype).cuda())
#                 else:
#                     regression_losses.append(torch.tensor(0).to(dtype))

#         # debug
#         imgs = kwargs.get('imgs', None)
#         if imgs is not None:
#             regressBoxes = BBoxTransform()
#             clipBoxes = ClipBoxes()
#             obj_list = kwargs.get('obj_list', None)
#             out = postprocess(imgs.detach(),
#                               torch.stack([anchors[0]] * imgs.shape[0], 0).detach(), regressions.detach(), classifications.detach(),
#                               regressBoxes, clipBoxes,
#                               0.5, 0.3)
#             imgs = imgs.permute(0, 2, 3, 1).cpu().numpy()
#             imgs = ((imgs * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255).astype(np.uint8)
#             imgs = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in imgs]
#             display(out, imgs, obj_list, imshow=False, imwrite=True)

#         return torch.stack(classification_losses).mean(dim=0, keepdim=True), \
#                torch.stack(regression_losses).mean(dim=0, keepdim=True) #* 50  # https://github.com/google/automl/blob/6fdd1de778408625c1faf368a327fe36ecd41bf7/efficientdet/hparams_config.py#L233


def reduce_loss(loss, reduction):
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".

    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()
    
def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Avarage factor when computing the mean of losses.

    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            loss = loss.sum() / avg_factor
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss