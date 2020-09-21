"""
最後更新時間：2020/09/20
更新內容
1. 有鑑於過去一口氣將裝有所有訓練資料的list用np.asarray轉成numpy格式會發生MemoryError，
   因此我在evaluate_model這個method加入「batch_process_slice_point」這個參數，可以使程式分批處理資料。
"""

### 載入函式庫 ###
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm

### Dice score 標準函式 ###
"""
### 參數說明 ###

target_roi_value = None: 該參數為目標的ROI(region of interest)值，預設為None，代表標記中只要出現大於1的值就視其為ROI；如果目標的ROI值為2(例如計算LiTS肝腫瘤的Dice)，就將該參數設成2；如果ROI目標值有1、3和5，就將該參數設成[1,3,5]

"""
def dice_score(gt, pd, target_roi_value = None): # gt = ground truth, pd = prediction
    if target_roi_value:
        roi_gt = np.isin(gt, target_roi_value)
        roi_pd = np.isin(pd, target_roi_value)
    else:
        roi_gt = np.greater(gt, 0)
        roi_pd = np.greater(pd, 0)
    try:
        if (roi_gt.sum() + roi_pd.sum()) > 0:
            dice = 2. * np.logical_and(roi_gt, roi_pd).sum() / (roi_gt.sum() + roi_pd.sum())
        else:
            dice = 0.0
    except ZeroDivisionError:
        dice = 0.0
    return dice

### 召回率：TP / (TP + FN) ###
def recall(gt, pd, target_roi_value = None):
    if target_roi_value:
        roi_gt = np.isin(gt, target_roi_value)
        roi_pd = np.isin(pd, target_roi_value)
    else:
        roi_gt = np.greater(gt, 0)
        roi_pd = np.greater(pd, 0)
    try:
        if roi_gt.sum() > 0:
            recall = np.logical_and(roi_gt, roi_pd).sum() / roi_gt.sum()
        else:
            recall = 0.0
    except ZeroDivisionError:
        recall = 0.0
    return recall

### 準確率：TP / (TP + FP) ### 
def precision(gt, pd, target_roi_value = None):
    if target_roi_value:
        roi_gt = np.isin(gt, target_roi_value)
        roi_pd = np.isin(pd, target_roi_value)
    else:
        roi_gt = np.greater(gt, 0)
        roi_pd = np.greater(pd, 0)
    try:
        if roi_pd.sum() > 0:
            precision = np.logical_and(roi_gt, roi_pd).sum() / roi_pd.sum()
        else:
            precision = 0.0
    except ZeroDivisionError:
        precision = 0.0
    return precision

### 生成模型預測結果的函式(適用於直接使用keras手刻模型的場合) ###
def predict_one_image(model, img_path, input_size, n_classes):
    
    img = cv2.imread(img_path) 
    img = cv2.resize(img, (input_size, input_size))
    
    pr_in = img.copy()
    pr_in = pr_in.astype(np.float32)
    pr_in[:, :, 0] -= 103.939 # B
    pr_in[:, :, 1] -= 116.779 # G
    pr_in[:, :, 2] -= 123.68 # R
    pr_in = pr_in[:, :, ::-1] 
    
    pr = model.predict(np.array([pr_in]))[0]
    pr = pr.reshape(input_size, input_size, n_classes)
    pr = pr.argmax(axis = 2)
    
    return pr
    
def predict_from_folder(model, inp_dir, input_size, n_classes):

    inps = glob.glob(os.path.join(inp_dir, "*.jpg")) + glob.glob(
        os.path.join(inp_dir, "*.png")) + \
        glob.glob(os.path.join(inp_dir, "*.jpeg"))
    inps = sorted(inps)

    assert type(inps) is list

    all_prs = []

    for i, inp in enumerate(tqdm(inps)):

        pr = predict_one_image(model, inp, input_size, n_classes)

        all_prs.append(pr)

    return all_prs

### 評估模型的函式 ###
"""
參數說明：
batch_process_slice_point = None: 分批處理的切點，因應有時無法一口氣將list轉成numpy的形式(資料量太大，發生MemoryError的問題)，需要分批處理資料，預設為None。如果將該參數設為[10000,20000]，代表將資料分成三批處理(第一批為第1到第10000筆資料；第二批為第10001到第20000筆資料；第三批為第20001以後的資料)。這裡推薦使用者以「病患」為單位去設置分批處理的切點，例如我想在前70位病患切一刀，就將數值設成前70位病患的影像總數(以KiTS19資料集為例，前70位病患的影像總數為15041。

target_roi_value = None: 該參數為目標的ROI(region of interest)值，預設為None，代表標記中只要出現大於1的值就視其為ROI；如果目標的ROI值為2(例如計算LiTS肝腫瘤的Dice)，就將該參數設成2；如果ROI目標值有1、3和5，就將該參數設成[1,3,5]
"""
def evaluate_model(image_dir, label_dir = None, checkpoints_path = None, calculate_predicting_indicators = True, output_predicted_result = False, segment_out_predicted_region_from_original_images = False, roi_description = 'roi', preds = None, batch_process_slice_point = None, target_roi_value = None):
    from keras_segmentation.predict import predict_multiple

    if preds is None:
        print('----------生成模型預測結果----------')
        preds = predict_multiple(checkpoints_path = checkpoints_path, inp_dir = image_dir)
        
    preds_batches = []
    
    if batch_process_slice_point:
        for i in range(len(batch_process_slice_point) + 1):
            if i == 0:
                preds_batches.append(preds[:batch_process_slice_point[i]])
            elif i == len(batch_process_slice_point):
                preds_batches.append(preds[batch_process_slice_point[i - 1]:])
            else:
                preds_batches.append(preds[batch_process_slice_point[i - 1]:batch_process_slice_point[i]])          
    else:
        preds_batches.append(preds)
    
    base_index = 0
    
    dice_score_list = []
    recall_list = []
    precision_list = []
    
    globalDice_part = []
    
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    
    counter1 = 0
    counter2 = 0
    counter3 = 0
    
    for idx, preds_batch in enumerate(preds_batches):     
        
        print(f'----------預測結果資料型態轉換(第{idx + 1}批資料)----------')
        preds_batch = np.asarray(preds_batch).astype(np.uint8)

        if calculate_predicting_indicators:
            labels_dicePerCase = []
            preds_dicePerCase = []

            labels_globalDice = []
            preds_globalDice = []

            # 取得該批資料第一位病患的編號 
            # 如果影像的命名方法為「病患編號_影像編號」，這裡的病患編號為「病患編號」
            # 如果影像的命名方法為「資料集_病患編號_影像編號」，這裡的病患編號為「資料集_病患編號」
            separator = '_'
            prev_patient_idx = separator.join(os.listdir(label_dir)[base_index + 0].split('_')[:-1])

            print(f'----------開始計算各項預測指標(第{idx + 1}批資料)----------')
            for i in range(preds_batch.shape[0]):

                label = cv2.imread(os.path.join(label_dir, os.listdir(label_dir)[base_index + i]), cv2.IMREAD_GRAYSCALE)    
                labels_globalDice.append(label)

                if preds_batch[i].shape != label.shape:
                    pred = cv2.resize(preds_batch[i].copy(), (label.shape[0],label.shape[1])) # 預測的標記照片大小必須和原始的標記照片一樣
                else:
                    pred = preds_batch[i].copy()
                preds_globalDice.append(pred)

                # 計算混淆矩陣
                if (1 in np.unique(label)) and (1 in np.unique(pred)):
                    TP += 1
                elif (1 not in np.unique(label)) and (1 in np.unique(pred)):
                    FP += 1
                elif (1 in np.unique(label)) and (1 not in np.unique(pred)):
                    FN += 1
                else:
                    TN += 1

                # 取得目前處理的影像對應的病患編號
                separator = '_'
                patient_idx = separator.join(os.listdir(label_dir)[base_index + i].split('_')[:-1])      

                if patient_idx != prev_patient_idx: # 判斷是否到達下一位病患的影像

                    labels_dicePerCase = np.asarray(labels_dicePerCase)
                    preds_dicePerCase = np.asarray(preds_dicePerCase)

                    # print(f'編號為{patient_idx}病患的CT影像張數：{labels_dicePerCase.shape[0]}') #####

                    dice_score_list.append(dice_score(labels_dicePerCase, preds_dicePerCase, target_roi_value))
                    recall_list.append(recall(labels_dicePerCase, preds_dicePerCase, target_roi_value))
                    precision_list.append(precision(labels_dicePerCase, preds_dicePerCase, target_roi_value))

                    labels_dicePerCase = []
                    preds_dicePerCase = []

                labels_dicePerCase.append(label)
                preds_dicePerCase.append(pred)

                if i == preds_batch.shape[0] - 1: # 判斷是否為最後一張的影像，如果是則開始計算最後一位病患的average Dice score per case
                    labels_dicePerCase = np.asarray(labels_dicePerCase)
                    preds_dicePerCase = np.asarray(preds_dicePerCase)

                    # print(f'編號為{patient_idx}病患的CT影像張數：{labels_dicePerCase.shape[0]}') #####

                    dice_score_list.append(dice_score(labels_dicePerCase, preds_dicePerCase, target_roi_value))
                    recall_list.append(recall(labels_dicePerCase, preds_dicePerCase, target_roi_value))
                    precision_list.append(precision(labels_dicePerCase, preds_dicePerCase, target_roi_value))                

                prev_patient_idx = patient_idx
                
                counter1 += 1
                if counter1 % 500 == 0:
                    print('目前進度：第' + str(counter1) + '張照片')

            ### 計算 global Dice (part) ###
            labels_globalDice = np.asarray(labels_globalDice)
            preds_globalDice = np.asarray(preds_globalDice)
            globalDice_part.append(dice_score(labels_globalDice, preds_globalDice, target_roi_value))

        if output_predicted_result:
            if label_dir:
                save_path = label_dir + '_predicted'
            else:
                save_path = image_dir.replace(image_dir.split('\\')[-1], 'annotations') + '_' + roi_description + '_predicted'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
                print('-----建立新資料夾：' + save_path + '-----')

            print(f'---------開始輸出模型預測結果(第{idx + 1}批資料)----------')
            
            for i in range(preds_batch.shape[0]):
                image_ori = cv2.imread(os.path.join(image_dir, os.listdir(image_dir)[base_index + i]), cv2.IMREAD_GRAYSCALE)
                image_ori_name = os.listdir(image_dir)[base_index + i]

                if preds_batch[i].shape != image_ori.shape:
                    pred = cv2.resize(preds_batch[i], (image_ori.shape[0],image_ori.shape[1]))
                else:
                    pred = preds_batch[i]

                cv2.imwrite(os.path.join(save_path, image_ori_name), pred)
                
                counter2 += 1
                if counter2 % 500 == 0:
                    print('目前進度：第' + str(counter2) + '張照片')

        if segment_out_predicted_region_from_original_images:
            save_path = image_dir + '_only_containing_predicted_roi_' + roi_description
            if not os.path.exists(save_path):
                os.makedirs(save_path)
                print('-----建立新資料夾：' + save_path + '-----')    


            print(f'----------開始生成並輸出只包含模型預測區域的圖片(第{idx + 1}批資料)----------')

            for i in range(preds_batch.shape[0]):
                image_ori = cv2.imread(os.path.join(image_dir, os.listdir(image_dir)[base_index + i]), cv2.IMREAD_GRAYSCALE)

                if preds_batch[i].shape != image_ori.shape:
                    pred = cv2.resize(preds_batch[i], (image_ori.shape[0],image_ori.shape[1]))
                else:
                    pred = preds_batch[i]

                image_pred_roi_region = pred * image_ori # 只保留原始圖片中模型預測區域的位置(其他區域視為像素等於0的背景)
                cv2.imwrite(os.path.join(save_path, os.listdir(image_dir)[base_index + i]), image_pred_roi_region)
                
                counter3 += 1
                if counter3 % 500 == 0:
                    print('目前進度：第' + str(counter3) + '張照片')
                
        base_index += len(preds_batch)
        
    ### 計算 global Dice ###
    if calculate_predicting_indicators:
        globalDice = np.mean(globalDice_part)

    print(f'total case number: {len(preds)}')

    if calculate_predicting_indicators:
        return np.mean(dice_score_list), np.mean(recall_list), np.mean(precision_list), globalDice, preds, dice_score_list, recall_list, precision_list, TP, FP, FN, TN
    else:
        return preds

### 將預測結果打印出來進行比較 ###
def show_result(target_dataset_base_dir, result_num = 5, roi_description = 'roi', roi_name_chinese = '重點區域',
    show_predicted_result = False, show_segmentation_result = False, image_scale = 4):
    
    for i in range(result_num):
        total_pic_num = 2
        pic_idx = 1
        if show_predicted_result:
            total_pic_num += 1
        if show_segmentation_result:
            total_pic_num += 1

        from random import choice

        target_img = choice(os.listdir(os.path.join(target_dataset_base_dir, 'images')))

        plt.rcParams['font.sans-serif'] = ['microsoft jhenghei']
        plt.subplots(figsize=(image_scale * 4, image_scale))

        plt.subplot(100 + 10 * total_pic_num + pic_idx)
        plt.title(f"原始CT影像\n({target_img})", fontsize = 6 + image_scale * 3)
        plt.imshow(
            cv2.imread(os.path.join(target_dataset_base_dir, 'images', target_img ), cv2.IMREAD_GRAYSCALE), 
            cmap = 'gray')

        pic_idx += 1
        plt.subplot(100 + 10 * total_pic_num + pic_idx)
        plt.title(f"{roi_name_chinese}標記\n({target_img})", fontsize = 6 + image_scale * 3)
        plt.imshow(
            cv2.imread(os.path.join(target_dataset_base_dir, f'annotations_{roi_description}', target_img), cv2.IMREAD_GRAYSCALE), 
            cmap = 'gray')

        if show_predicted_result:
            pic_idx += 1
            plt.subplot(100 + 10 * total_pic_num + pic_idx)
            plt.title(f"模型預測的{roi_name_chinese}標記\n({target_img})", fontsize = 6 + image_scale * 3)
            plt.imshow(
                cv2.imread(
                    os.path.join(target_dataset_base_dir, f'annotations_{roi_description}_predicted', target_img), 
                    cv2.IMREAD_GRAYSCALE), 
                cmap = 'gray')

        if show_segmentation_result:
            pic_idx += 1
            plt.subplot(100 + 10 * total_pic_num + pic_idx)
            plt.title(f"模型切割出來的{roi_name_chinese}影像\n({target_img})", fontsize = 6 + image_scale * 3)
            plt.imshow(
                cv2.imread(
                    os.path.join(target_dataset_base_dir, f'images_only_containing_predicted_roi_{roi_description}', target_img), 
                    cv2.IMREAD_GRAYSCALE), 
                cmap = 'gray')
