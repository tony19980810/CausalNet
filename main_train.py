import random
from os import path
import os
import cv2
import time

import pandas
from sklearn.metrics import confusion_matrix
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import argparse
from distutils.util import strtobool
import torch

import numpy as np


from Models import CausalNet


def reset_weights(m):
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):

            layer.reset_parameters()

def confusionMatrix(gt, pred, show=False):
    TN, FP, FN, TP = confusion_matrix(gt, pred).ravel()
    f1_score = (2 * TP) / (2 * TP + FP + FN)
    num_samples = len([x for x in gt if x == 1])
    average_recall = TP / num_samples
    return f1_score, average_recall


def recognition_evaluation(final_gt, final_pred, show=False):
    label_dict = {'Negative': 0, 'Positive': 1, 'surprise': 2}
    # Display recognition result
    f1_list = []
    ar_list = []
    try:
        for emotion, emotion_index in label_dict.items():
            gt_recog = [1 if x == emotion_index else 0 for x in final_gt]
            pred_recog = [1 if x == emotion_index else 0 for x in final_pred]
            try:
                f1_recog, ar_recog = confusionMatrix(gt_recog, pred_recog)
                f1_list.append(f1_recog)
                ar_list.append(ar_recog)
            except Exception as e:
                pass
        UF1 = np.mean(f1_list)
        UAR = np.mean(ar_list)
        return UF1, UAR
    except:
        return '', ''



def whole_face_block_coordinates():
    df = pandas.read_csv('combined_3_class2_for_optical_flow.csv')
    m, n = df.shape

    face_block_coordinates = {}


    for i in range(0, m):
        image_name = str(df['sub'][i]) + '_' + str(
            df['filename_o'][i]) + ' .png'

        batch_landmarks=None

        if batch_landmarks is None:

            batch_landmarks = np.array([[[9.528073, 11.062551]
                                            , [21.396168, 10.919773]
                                            , [15.380184, 17.380562]
                                            , [10.255435, 22.121233]
                                            , [20.583706, 22.25584]]])

        row_n, col_n = np.shape(batch_landmarks[0])

        for i in range(0, row_n):
            for j in range(0, col_n):
                if batch_landmarks[0][i][j] < 7:
                    batch_landmarks[0][i][j] = 7
                if batch_landmarks[0][i][j] > 21:
                    batch_landmarks[0][i][j] = 21


        batch_landmarks = batch_landmarks.astype(int)

        face_block_coordinates[image_name] = batch_landmarks[0]
        tmp=image_name.split(' ')[0]
        tmp1=tmp+'_1 .png'
        face_block_coordinates[tmp1] = batch_landmarks[0]
        tmp2 = tmp + '_2 .png'
        face_block_coordinates[tmp2] = batch_landmarks[0]
        tmp3 = tmp + '_3 .png'
        face_block_coordinates[tmp3] = batch_landmarks[0]

    return face_block_coordinates


def crop_optical_flow_block():
    face_block_coordinates_dict = whole_face_block_coordinates()

    whole_optical_flow_path = './datasets/STSNet_whole_norm_u_v_os'
    whole_optical_flow_imgs = os.listdir(whole_optical_flow_path)
    four_parts_optical_flow_imgs = {}

    for n_img in whole_optical_flow_imgs:
        four_parts_optical_flow_imgs[n_img]=[]
        flow_image = cv2.imread(whole_optical_flow_path + '/' + n_img)
        four_part_coordinates = face_block_coordinates_dict[n_img]
        l_eye = flow_image[four_part_coordinates[0][0]-7:four_part_coordinates[0][0]+7,
                four_part_coordinates[0][1]-7: four_part_coordinates[0][1]+7]
        l_lips = flow_image[four_part_coordinates[1][0] - 7:four_part_coordinates[1][0] + 7,
                four_part_coordinates[1][1] - 7: four_part_coordinates[1][1] + 7]
        nose = flow_image[four_part_coordinates[2][0] - 7:four_part_coordinates[2][0] + 7,
                four_part_coordinates[2][1] - 7: four_part_coordinates[2][1] + 7]
        r_eye = flow_image[four_part_coordinates[3][0] - 7:four_part_coordinates[3][0] + 7,
                four_part_coordinates[3][1] - 7: four_part_coordinates[3][1] + 7]
        r_lips = flow_image[four_part_coordinates[4][0] - 7:four_part_coordinates[4][0] + 7,
                four_part_coordinates[4][1] - 7: four_part_coordinates[4][1] + 7]
        four_parts_optical_flow_imgs[n_img].append(l_eye)
        four_parts_optical_flow_imgs[n_img].append(l_lips)
        four_parts_optical_flow_imgs[n_img].append(nose)
        four_parts_optical_flow_imgs[n_img].append(r_eye)
        four_parts_optical_flow_imgs[n_img].append(r_lips)

    return four_parts_optical_flow_imgs



def main(config):
    seed = 2025
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    learning_rate = 0.00005
    batch_size = 256*4
    epochs = 200
    all_accuracy_dict = {}
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    loss_fn = nn.CrossEntropyLoss()
    if (config.train):
        if not path.exists('ourmodel_threedatasets_weights'):
            os.mkdir('ourmodel_threedatasets_weights')

    print('lr=%f, epochs=%d, device=%s\n' % (learning_rate, epochs, device))

    total_gt = []
    total_pred = []
    best_total_pred = []

    t = time.time()

    main_path = './datasets/three_norm_u_v_os'
    subName = os.listdir(main_path)
    all_five_parts_optical_flow = crop_optical_flow_block()
    print(subName)

    for n_subName in subName:
        print('Subject:', n_subName)
        y_train = []
        y_test = []
        four_parts_train = []

        four_parts_test = []

        # Get train dataset
        expression = os.listdir(main_path + '/' + n_subName + '/u_train')
        for n_expression in expression:
            img = os.listdir(main_path + '/' + n_subName + '/u_train/' + n_expression)

            for n_img in img:
                y_train.append(int(n_expression))

                l_eye_lips = cv2.hconcat([all_five_parts_optical_flow[n_img][0], all_five_parts_optical_flow[n_img][1]])
                r_eye_lips  =  cv2.hconcat([all_five_parts_optical_flow[n_img][3], all_five_parts_optical_flow[n_img][4]])
                lr_eye_lips = cv2.vconcat([l_eye_lips, r_eye_lips])

                n_img1=n_img.split(' ')[0]+'_1 .png'

                l_eye_lips1 = cv2.hconcat([all_five_parts_optical_flow[n_img1][0], all_five_parts_optical_flow[n_img1][1]])
                r_eye_lips1 = cv2.hconcat([all_five_parts_optical_flow[n_img1][3], all_five_parts_optical_flow[n_img1][4]])
                lr_eye_lips1 = cv2.vconcat([l_eye_lips1, r_eye_lips1])

                n_img2 = n_img.split(' ')[0] + '_2 .png'

                l_eye_lips2 = cv2.hconcat(
                    [all_five_parts_optical_flow[n_img2][0], all_five_parts_optical_flow[n_img2][1]])
                r_eye_lips2 = cv2.hconcat(
                    [all_five_parts_optical_flow[n_img2][3], all_five_parts_optical_flow[n_img2][4]])
                lr_eye_lips2 = cv2.vconcat([l_eye_lips2, r_eye_lips2])

                n_img3 = n_img.split(' ')[0] + '_3 .png'

                l_eye_lips3 = cv2.hconcat(
                    [all_five_parts_optical_flow[n_img3][0], all_five_parts_optical_flow[n_img3][1]])
                r_eye_lips3 = cv2.hconcat(
                    [all_five_parts_optical_flow[n_img3][3], all_five_parts_optical_flow[n_img3][4]])
                lr_eye_lips3 = cv2.vconcat([l_eye_lips3, r_eye_lips3])
              
                four_parts_train.append([lr_eye_lips,lr_eye_lips1,lr_eye_lips2,lr_eye_lips3])






        # Get test dataset
        expression = os.listdir(main_path + '/' + n_subName + '/u_test')
        for n_expression in expression:
            img = os.listdir(main_path + '/' + n_subName + '/u_test/' + n_expression)

            for n_img in img:
                y_test.append(int(n_expression))
                l_eye_lips = cv2.hconcat([all_five_parts_optical_flow[n_img][0], all_five_parts_optical_flow[n_img][1]])
                r_eye_lips = cv2.hconcat([all_five_parts_optical_flow[n_img][3], all_five_parts_optical_flow[n_img][4]])
                lr_eye_lips = cv2.vconcat([l_eye_lips, r_eye_lips])

                n_img1 = n_img.split(' ')[0] + '_1 .png'

                l_eye_lips1 = cv2.hconcat(
                    [all_five_parts_optical_flow[n_img1][0], all_five_parts_optical_flow[n_img1][1]])
                r_eye_lips1 = cv2.hconcat(
                    [all_five_parts_optical_flow[n_img1][3], all_five_parts_optical_flow[n_img1][4]])
                lr_eye_lips1 = cv2.vconcat([l_eye_lips1, r_eye_lips1])

                n_img2 = n_img.split(' ')[0] + '_2 .png'

                l_eye_lips2 = cv2.hconcat(
                    [all_five_parts_optical_flow[n_img2][0], all_five_parts_optical_flow[n_img2][1]])
                r_eye_lips2 = cv2.hconcat(
                    [all_five_parts_optical_flow[n_img2][3], all_five_parts_optical_flow[n_img2][4]])
                lr_eye_lips2 = cv2.vconcat([l_eye_lips2, r_eye_lips2])

                n_img3 = n_img.split(' ')[0] + '_3 .png'

                l_eye_lips3 = cv2.hconcat(
                    [all_five_parts_optical_flow[n_img3][0], all_five_parts_optical_flow[n_img3][1]])
                r_eye_lips3 = cv2.hconcat(
                    [all_five_parts_optical_flow[n_img3][3], all_five_parts_optical_flow[n_img3][4]])
                lr_eye_lips3 = cv2.vconcat([l_eye_lips3, r_eye_lips3])

                four_parts_test.append([lr_eye_lips, lr_eye_lips1,lr_eye_lips2,lr_eye_lips3])

        # weight_path = 'I:\ourmodel_threedatasets_weights' + '/' + n_subName + '.pth'


        model = CausalNet(
            image_size=28,
            patch_size=7,
            dim=256,  # 256,--96, 56-, 192
            heads=3,  # 3 ---- , 6-
            num_hierarchies=3,  # 3----number of hierarchies
            block_repeats=(3, 3, 9),  # (2, 2, 8),------

            num_classes=3,
            gamma=0.5
        )
        model = model.to(device)

        if(config.train):

            print('train')
            print(len(y_train))
            print(len(y_test))
        # else:
        #     model.load_state_dict(torch.load(weight_path))
        optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
        y_train = torch.Tensor(y_train).to(dtype=torch.long)

        four_parts_train =  torch.Tensor(np.array(four_parts_train))

        four_parts_train =four_parts_train.permute(0, 1,4, 2, 3)


        dataset_train = TensorDataset(four_parts_train, y_train)
        train_dl = DataLoader(dataset_train, batch_size=batch_size)
        y_test = torch.Tensor(y_test).to(dtype=torch.long)
        four_parts_test = torch.Tensor(np.array(four_parts_test))
        four_parts_test=four_parts_test.permute(0, 1,4, 2, 3)
        dataset_test = TensorDataset(four_parts_test, y_test)
        test_dl = DataLoader(dataset_test, batch_size=batch_size)
        # store best results
        best_accuracy_for_each_subject = 0
        best_each_subject_pred = []
        best_epoch=0
        best_matrix=[]
        for epoch in range(1, epochs + 1):
            if (config.train):
                # Training
                model.train()
                train_loss = 0.0
                num_train_correct = 0
                num_train_examples = 0
                matrix=[]

                for batch in train_dl:
                    optimizer.zero_grad()
                    x = batch[0].to(device)

                    y = batch[1].to(device)
                    yhat = model(x)
                    loss = loss_fn(yhat, y)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.data.item() * x.size(0)
                    num_train_correct += (torch.max(yhat, 1)[1] == y).sum().item()
                    num_train_examples += x.shape[0]




            model.eval()
            val_loss = 0.0
            num_val_correct = 0
            num_val_examples = 0
            for batch in test_dl:
                x = batch[0].to(device)
                y = batch[1].to(device)
                yhat = model(x)
                loss = loss_fn(yhat, y)

                _, predicts = torch.max(yhat, 1)
                for a in range(0, len(predicts)):
                    matrix.append([int(predicts[a]), int(y[a])])

                val_loss += loss.data.item() * x.size(0)
                num_val_correct += (torch.max(yhat, 1)[1] == y).sum().item()
                num_val_examples += y.shape[0]

            val_acc = num_val_correct / num_val_examples
            val_loss = val_loss / len(test_dl.dataset)
            print("[Epoch %d] Validation accuracy:%.4f. Loss:%.3f" % (epoch, val_acc, val_loss))


            temp_best_each_subject_pred = []
            if best_accuracy_for_each_subject < val_acc:
                best_accuracy_for_each_subject = val_acc
                temp_best_each_subject_pred.extend(torch.max(yhat, 1)[1].tolist())
                best_each_subject_pred = temp_best_each_subject_pred
                best_matrix=matrix
                best_epoch=epoch
                # Save Weights
                # if (config.train):
                #     torch.save(model.state_dict(), weight_path)
            if val_acc>=1:
                if not os.path.exists(os.path.join('.', 'results')):
                    os.makedirs(os.path.join('.', 'results'))
                with open(os.path.join('.', 'results', str(n_subName) + '_acc.txt'), 'a') as f:
                    f.write('best epoach: ' + str(best_epoch) + '\n' + 'best acc: ' + str(
                        best_accuracy_for_each_subject) + '\n' + 'matrix_acc: ' + str(best_matrix) + '\n')

                break
            if epoch == epochs:

                if not os.path.exists(os.path.join('.', 'results')):
                    os.makedirs(os.path.join('.', 'results'))
                with open(os.path.join('.', 'results', str(n_subName) + '_acc.txt'), 'a') as f:
                    f.write('best epoach: ' + str(best_epoch) + '\n' + 'best acc: ' + str(
                        best_accuracy_for_each_subject) + '\n' + 'matrix_acc: ' + str(best_matrix) + '\n')



        print('Best Predicted    :', best_each_subject_pred)
        accuracydict = {}
        accuracydict['pred'] = best_each_subject_pred
        accuracydict['truth'] = y.tolist()
        all_accuracy_dict[n_subName] = accuracydict

        print('Ground Truth :', y.tolist())
        print('Evaluation until this subject: ')
        total_pred.extend(torch.max(yhat, 1)[1].tolist())
        total_gt.extend(y.tolist())
        best_total_pred.extend(best_each_subject_pred)
        UF1, UAR = recognition_evaluation(total_gt, total_pred, show=True)
        best_UF1, best_UAR = recognition_evaluation(total_gt, best_total_pred, show=True)
        print('best UF1:', round(best_UF1, 4), '| best UAR:', round(best_UAR, 4))

    print('Final Evaluation: ')
    UF1, UAR = recognition_evaluation(total_gt, total_pred)
    print(np.shape(total_gt))
    print('Total Time Taken:', time.time() - t)
    print(all_accuracy_dict)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--train', type=strtobool, default=True)
    config = parser.parse_args()
    main(config)
