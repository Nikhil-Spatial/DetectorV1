from configs import S, C, B, IDX_TO_CLASS, CONFIDENCE_THRESHOLD

def decode_preds(preds_batch):
    decoded_preds = []

    for pred in preds_batch:
        pred = pred.reshape((S, S, C + B * 5))
        objects = []

        for i in range(S):
            for j in range(S):
                pred_cell = pred[i][j]

                pred_class_idx = pred_cell[:20].argmax().item()
                pred_class_prob = pred_cell[pred_class_idx].item()
                pred_class = IDX_TO_CLASS[pred_class_idx]

                pred_1_confidence = (pred_cell[24].item() * \
                                     pred_class_prob,)
                pred_2_confidence = (pred_cell[29].item() * \
                                     pred_class_prob,)

                bbox_1 = convert_xywh_coords(pred_cell[20:24], i, j, False,
                                             True)
                bbox_2 = convert_xywh_coords(pred_cell[25:29], i, j, False,
                                             True)

                objects.append((pred_class,) + pred_1_confidence + bbox_1)
                objects.append((pred_class,) + pred_2_confidence + bbox_2)

        decoded_preds.append(objects)

    return decoded_preds

def confidence_threshold(decoded_preds):
    valid_preds = []

    for image in decoded_preds:
        image_preds = []
        for pred in image:
            if pred[1] > CONFIDENCE_THRESHOLD:
                image_preds.append(pred)

        if image_preds:
            valid_preds.append(image_preds)
            
    return valid_preds