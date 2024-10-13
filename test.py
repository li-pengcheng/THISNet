# *_*coding:utf-8 *_*
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import shutil
import torch
from tqdm import tqdm
from dataloader.dataio import PlyDataSet_test
from networks.THISNet import THISNet
from torch.utils.data import DataLoader
from dataloader.dataio import generate_plyfile


def knn_ind(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k+1, dim=-1)[1]  # (batch_size, num_points, k)
    return idx



def test_semseg(model, loader, num_classes = 8, gpu=True, generate_ply=False, confusion_matrics=False, save_dir=None):

    with torch.no_grad():
        for batch_id, meshdata in tqdm(enumerate(loader,0), total=len(loader), smoothing=0.9):
            faces, faces_points, faces_center_points, faces_normals, faces_center_normal, names = meshdata
            data = torch.cat((faces_center_points, faces_points, faces_center_normal, faces_normals), dim=2)
            data = data.transpose(2, 1)

            batchsize, _, num_point = data.size()

            output, pred = model(data.float().cuda())
            for item in range(len(output["pred_masks"])):
                pred_scores = output["pred_logits"][item].sigmoid()
                pred_masks = output["pred_masks"][item].sigmoid()
                pred_objectness = output["pred_scores"][item].sigmoid()
                pred_scores = torch.sqrt(pred_scores * pred_objectness)
                # max/argmax
                scores, labels = pred_scores.max(dim=-1)
                # cls threshold
                keep = scores > 0.5
                scores = scores[keep]
                labels = labels[keep]
                mask_pred_per_image = pred_masks[keep]
                pred_masks = mask_pred_per_image > 0.5

                index = torch.where(pred_masks.sum(0) > 1) # overlay points
                pred_masks[:,index[0].cpu().numpy()] = 0

                pred_label = (pred_masks * (labels[:, None] + 1)).sum(0)

                if generate_ply:
                    name = names[item].split(".")[0] + ".ply"
                    face = faces[item].numpy()
                    point_face = faces_points[item].numpy()
                    face_normal = faces_normals[item].numpy()
                    generate_plyfile(face, point_face, face_normal, pred_label.cpu().numpy(), os.path.join(save_dir, name))


if __name__== "__main__":
    """
    ---------------------------------------settings--------------------------------------------------------
    """
    test_plypath = 'xxx'
    checkpoint = torch.load('ckpt_xx_xxxx.pth')
    save_dir = 'xxx'
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)
    """
    ---------------------------------------models--------------------------------------------------------
    """
    num_classes = 14
    model = THISNet(in_channels=24, output_channels=num_classes) # coord or normal channels
    model = torch.nn.DataParallel(model, device_ids=[0])
    model.cuda()

    model.load_state_dict(checkpoint)

    test_dataset = PlyDataSet_test(plyPath=test_plypath)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=4)

    test_semseg(model, test_loader, num_classes=num_classes, generate_ply=True,
                                         confusion_matrics=True,
                                         save_dir=str(save_dir))

    print('done!')







