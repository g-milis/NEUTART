"""
crop
for torch tensor
Given image, bbox(center, bboxsize)
return: cropped image, tform(used for transform the keypoint accordingly)
only support crop to squared images
"""
import torch


def transform_points(points, tform, points_scale=None, out_scale=None):
    points_2d = points[:,:,:2]
        
    # Input points must use original range
    if points_scale:
        assert points_scale[0] == points_scale[1]
        points_2d = (points_2d * 0.5 + 0.5) * points_scale[0]

    batch_size, n_points, _ = points.shape
    trans_points_2d = torch.bmm(
        torch.cat([points_2d, torch.ones([batch_size, n_points, 1], device=points.device, dtype=points.dtype)], dim=-1), 
        tform
    ) 
    if out_scale: # h,w of output image size
        trans_points_2d[:, :, 0] = trans_points_2d[:,:,0] / out_scale[1] * 2 - 1
        trans_points_2d[:, :, 1] = trans_points_2d[:,:,1] / out_scale[0] * 2 - 1
    trans_points = torch.cat([trans_points_2d[:, :, :2], points[:, :, 2:]], dim=-1)
    return trans_points
