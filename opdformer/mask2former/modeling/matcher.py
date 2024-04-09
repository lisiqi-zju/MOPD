# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/matcher.py
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.cuda.amp import autocast

from detectron2.projects.point_rend.point_features import point_sample


def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


batch_dice_loss_jit = torch.jit.script(
    batch_dice_loss
)  # type: torch.jit.ScriptModule


def batch_sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    hw = inputs.shape[1]

    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )

    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum(
        "nc,mc->nm", neg, (1 - targets)
    )

    return loss / hw


batch_sigmoid_ce_loss_jit = torch.jit.script(
    batch_sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_mask: float = 1, cost_dice: float = 1, num_points: int = 0):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_mask: This is the relative weight of the focal loss of the binary mask in the matching cost
            cost_dice: This is the relative weight of the dice loss of the binary mask in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice

        assert cost_class != 0 or cost_mask != 0 or cost_dice != 0, "all costs cant be 0"

        self.num_points = num_points

    @torch.no_grad()
    def memory_efficient_forward(self, outputs, targets):
        """More memory-friendly matching"""
        bs, num_queries = outputs["pred_logits"].shape[:2]

        indices = []
        indices2 = []
        ############################################
        def pad_and_concat(tensor_list):  

            # 确定tensor列表中的最大长度  

            max_len = max(tensor.size(0) for tensor in tensor_list)  

            

            # 初始化一个空的二维tensor列表  

            padded_tensors = []  

            

            # 初始化一个空的mask tensor列表  

            masks = []  

            

            # 遍历每个tensor进行填充  

            for tensor in tensor_list:  

                # 计算需要填充的长度  

                padding_len = max_len - tensor.size(0)  

                

                # 在tensor的开头进行填充，使用tensor中的最后一个元素作为填充值  

                # 也可以使用0或其他默认值作为填充值  

                padded_tensor = torch.nn.functional.pad(tensor, (0, padding_len), 'constant', value=tensor[-1])  

                

                # 创建一个mask tensor，原始tensor部分填充为1，填充部分填充为0  

                mask = torch.ones(padded_tensor.size(), dtype=torch.bool)  

                mask[:tensor.size(0)] = True  

                mask[tensor.size(0):] = False  

                

                # 将填充后的tensor和mask添加到列表中  

                padded_tensors.append(padded_tensor)  

                masks.append(mask)  

            

            # 将所有填充后的tensor拼接成一个二维tensor  

            concatenated_tensor = torch.stack(padded_tensors)  

            

            # 将所有mask tensor拼接成一个二维tensor  

            concatenated_mask = torch.stack(masks)  

            

            return concatenated_tensor, concatenated_mask
        
        def pad_box_tensor(box_list):  

            """  

            Pads the box tensors in box_list to the maximum length of any box tensor.  

            

            Args:  

                box_list (list of torch.Tensor): List of box tensors, where each tensor has shape (N, 4)  

                

            Returns:  

                padded_box_tensor (torch.Tensor): Padded box tensor of shape (len(box_list), max_num_boxes, 4)  

            """  

            # 计算box_list中所有box tensor的最大长度  

            max_num_boxes = max(box_tensor.size(0) for box_tensor in box_list)  

            

            # 创建一个空的tensor来存放填充后的box tensor  

            padded_box_tensor = torch.full((len(box_list), max_num_boxes, 4), -1, dtype=torch.float32)  # 用-1作为填充值  

            

            # 遍历每个box tensor，并将其复制到填充tensor的对应位置  

            for i, box_tensor in enumerate(box_list):  

                num_boxes = box_tensor.size(0)  

                padded_box_tensor[i, :num_boxes] = box_tensor  

            

            return padded_box_tensor 
                    
        # def
        
        ##########################################
        # cost_matrix= matching_method.compute_cost_matrix(outputs,outputs,None)
        # Iterate through batch size
        import torch
        for b in range(bs):

            out_prob = outputs["pred_logits"][b].softmax(-1)  # [num_queries, num_classes]
            tgt_ids = targets[b]["labels"]

            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            cost_class = -out_prob[:, tgt_ids]

            out_mask = outputs["pred_masks"][b]  # [num_queries, H_pred, W_pred]
            # gt masks are already padded when preparing target
            tgt_mask = targets[b]["masks"].to(out_mask)

            out_mask = out_mask[:, None]
            tgt_mask = tgt_mask[:, None]
            # all masks share the same set of points for efficient matching!
            point_coords = torch.rand(1, self.num_points, 2, device=out_mask.device)
            # get gt labels
            tgt_mask = point_sample(
                tgt_mask,
                point_coords.repeat(tgt_mask.shape[0], 1, 1),
                align_corners=False,
            ).squeeze(1)

            out_mask = point_sample(
                out_mask,
                point_coords.repeat(out_mask.shape[0], 1, 1),
                align_corners=False,
            ).squeeze(1)

            with autocast(enabled=False):
                out_mask = out_mask.float()
                tgt_mask = tgt_mask.float()
                # Compute the focal loss between masks
                if out_mask.shape[0] == 0 or tgt_mask.shape[0] == 0:
                    cost_mask = batch_sigmoid_ce_loss(out_mask, tgt_mask)
                    # Compute the dice loss betwen masks
                    cost_dice = batch_dice_loss(out_mask, tgt_mask)
                else:
                    cost_mask = batch_sigmoid_ce_loss_jit(out_mask, tgt_mask)
                    # Compute the dice loss betwen masks
                    cost_dice = batch_dice_loss_jit(out_mask, tgt_mask)
            # Final cost matrix
            C = (
                self.cost_mask * cost_mask
                + self.cost_class * cost_class
                + self.cost_dice * cost_dice
            )
            C = C.reshape(num_queries, -1).cpu()

            indices.append(linear_sum_assignment(C))
            
            from detectron2.structures import BitMasks
            import torch
            assert "pred_logits" in outputs
            pred_logits = torch.unsqueeze(outputs["pred_logits"][b],dim=0).float()
            pred_masks = torch.unsqueeze(outputs["pred_masks"][b],dim=0).float()
            
            pred_masks=F.interpolate(
                    pred_masks,
                    size=(targets[0]["masks"].shape[-2], targets[0]["masks"].shape[-1]),
                    mode="bilinear",
                    align_corners=False,
                )
                    
            pred_boxes = [torch.unsqueeze(BitMasks(mask_pred> 0).get_bounding_boxes().tensor,dim=0) for mask_pred in pred_masks]
            pred_boxes = torch.cat(pred_boxes,dim=0)
            
            pred = {"pred_logits": pred_logits.to(outputs["pred_logits"].device),
                    "pred_boxes": pred_boxes.to(outputs["pred_logits"].device)}
            # targets_labels = [torch.unsqueeze(target["labels"],dim=0) for target in targets]
            targets_labels = torch.unsqueeze(targets[b]["labels"],dim=0)
            targets_boxes = torch.unsqueeze(targets[b]["gt_bbox"].tensor ,dim=0)
            # targets_boxes = [target["gt_bbox"].tensor for target in targets]
            targets_labels,targets_masks = pad_and_concat(targets_labels) 
            targets_boxes=pad_box_tensor(targets_boxes)
            
            # targets_masks = torch.cat([target["masks"] for target in targets],dim=0) 
            # a=BitMasks(targets_masks[0] > 0).get_bounding_boxes()
            # b=BitMasks(outputs["pred_masks"][0]>0).get_bounding_boxes()
            # a=BitMasks(targets_masks[0]>0).get_bounding_boxes()
            # targets_boxes = [torch.unsqueeze(BitMasks(mask > 0).get_bounding_boxes().tensor,dim=0) for mask in targets_masks]
            # targets_boxes = torch.cat(targets_boxes,dim=0)
            target={"labels": targets_labels.to(outputs["pred_logits"].device),
                    "boxes":targets_boxes.to(outputs["pred_logits"].device),
                    "mask":targets_masks.to(outputs["pred_logits"].device)}
            
            import torch
            from torch.nn import L1Loss, CrossEntropyLoss

            from uotod.match import BalancedSinkhorn,Hungarian
            from uotod.loss import DetectionLoss
            from uotod.loss import MultipleObjectiveLoss, GIoULoss, NegativeProbLoss

            # matching_method=Hungarian(
            # loc_match_module=GIoULoss(reduction="none"),
            # background_cost=0.,)
            matching_method = BalancedSinkhorn(
                cls_match_module=NegativeProbLoss(reduction="none"),
                loc_match_module=MultipleObjectiveLoss(
                    losses=[GIoULoss(reduction="none"), L1Loss(reduction="none")],
                    weights=[1., 5.],
                ),
                background_cost=0.,  # Does not influence the matching when using balanced OT
            )
            matching = matching_method(pred, target, None)
            cost_matrix= matching_method.compute_cost_matrix(pred,target,None)
            matching = matching_method.compute_matching(cost_matrix, target["mask"])
            matching = matching.squeeze(0)
            max_values, max_indices = torch.max(matching, dim=0)  
            row_indices = torch.arange(matching.size(0)).unsqueeze(1).to(matching.device) 
            max_positions = torch.cat((max_indices.unsqueeze(1).to(matching.device) ,torch.arange(0, max_indices.shape[0]).unsqueeze(1).to(matching.device) ), dim=1) 
            indices_to_remove = max_positions[:, 1] == C.shape[1] 
            max_positions_filtered = max_positions[~indices_to_remove] 
            output = max_positions_filtered.unbind(dim=1)
            
            if torch.isnan(matching).any():
                continue
            else:
                indices2.append(output)  
        return indices2
        # if torch.isnan(matching).any():
        #     return [
        #         (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
        #         for i, j in indices
        #     ]
        # else:
        #     return [
        #         (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
        #         for i, j in indices2
        #     ]
            

    @torch.no_grad()
    def forward(self, outputs, targets):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_masks": Tensor of dim [batch_size, num_queries, H_pred, W_pred] with the predicted masks

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "masks": Tensor of dim [num_target_boxes, H_gt, W_gt] containing the target masks

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        return self.memory_efficient_forward(outputs, targets)

    def __repr__(self, _repr_indent=4):
        head = "Matcher " + self.__class__.__name__
        body = [
            "cost_class: {}".format(self.cost_class),
            "cost_mask: {}".format(self.cost_mask),
            "cost_dice: {}".format(self.cost_dice),
        ]
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
