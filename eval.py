import os
import random
import torch
import numpy as np
import cv2
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score

from Models import CausalNet
from main_train import crop_optical_flow_block


subject_list = ['sub17', 'sub26', 'sub16', 'sub09', 'sub05', 'sub24', 'sub02', 'sub13', 'sub04', 'sub23',
                'sub11', 'sub12', 'sub08', 'sub14', 'sub03', 'sub19', 'sub01',
                'sub20', 'sub21', 'sub22', 'sub15', 'sub06', 'sub25', 'sub07', '006', '007', '009', '010',
                '011', '012', '013', '014', '015', '016', '017', '018', '019', '020', '021', '022', '023',
                '024', '026', '028', '030', '032', '031', '033', '034', '035', '036', '037', 's01', 's02',
                's03', 's04', 's05', 's06', 's08', 's09', 's11', 's12', 's13', 's14', 's15', 's18', 's19',
                's20']
gamma = 0.5
layer1, layer2, layer3 = 3, 3, 9
main_path = './datasets/three_norm_u_v_os'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 64


seed = 2025
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


print("Loading precomputed optical flow blocks...")
all_five_parts_optical_flow = crop_optical_flow_block()
print("Loaded.")


for subject_name in subject_list:
    weight_path = f'I:\ourmodel_threedatasets_weights/{subject_name}.pth'
    if not os.path.exists(weight_path):
        print(f"[Skipped] Weight not found for subject {subject_name}")
        continue

    print(f"\n===== Testing Subject: {subject_name} =====")


    model = CausalNet(
        image_size=28,
        patch_size=7,
        dim=256,
        heads=3,
        num_hierarchies=3,
        block_repeats=(layer1, layer2, layer3),
        num_classes=3,
        gamma=gamma
    ).to(device)
    model.load_state_dict(torch.load(weight_path))
    model.eval()


    y_test = []
    four_parts_test = []
    subject_path = os.path.join(main_path, subject_name, 'u_test')
    if not os.path.exists(subject_path):
        print(f"[Skipped] Test data not found for subject {subject_name}")
        continue

    expressions = os.listdir(subject_path)
    for n_expression in expressions:
        img_list = os.listdir(os.path.join(subject_path, n_expression))
        for n_img in img_list:
            y_test.append(int(n_expression))
            imgs = []
            for i in range(4):
                suffix = f'_{i} .png' if i > 0 else ' .png'
                name = n_img.split(' ')[0] + suffix
                if name not in all_five_parts_optical_flow:
                    print(f"[Warning] Missing optical flow image: {name}")
                    continue
                l_eye_lips = cv2.hconcat([all_five_parts_optical_flow[name][0], all_five_parts_optical_flow[name][1]])
                r_eye_lips = cv2.hconcat([all_five_parts_optical_flow[name][3], all_five_parts_optical_flow[name][4]])
                imgs.append(cv2.vconcat([l_eye_lips, r_eye_lips]))
            if len(imgs) == 4:
                four_parts_test.append(imgs)

    if len(four_parts_test) == 0:
        print(f"[Skipped] No valid test samples for {subject_name}")
        continue


    y_test_tensor = torch.Tensor(y_test[:len(four_parts_test)]).to(dtype=torch.long)
    four_parts_test_tensor = torch.Tensor(np.array(four_parts_test)).permute(0, 1, 4, 2, 3)

    test_dataset = TensorDataset(four_parts_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)


    all_preds = []
    all_gts = []
    matrix = []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            yhat = model(x)
            _, preds = torch.max(yhat, 1)
            for a in range(0, len(preds)):
                matrix.append([int(preds[a]), int(y[a])])
            all_preds.extend(preds.cpu().numpy())
            all_gts.extend(y.cpu().numpy())


    acc = accuracy_score(all_gts, all_preds)

    if not os.path.exists(os.path.join('.', 'results_eval')):
        os.makedirs(
            os.path.join('.', 'results_eval'))
    with open(os.path.join('.', 'results_eval',
                           str(subject_name) + '_acc.txt'),
              'a') as f:
        f.write('best epoach: ' + str(1) + '\n' + 'best acc: ' + str(
            acc) + '\n' + 'matrix_acc: ' + str(
            matrix) + '\n')
