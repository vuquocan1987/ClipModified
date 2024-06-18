# model_path
import os
import sys
import torch.nn.functional as F
from monai.inferers import sliding_window_inference, SlidingWindowSplitter

os.chdir('/CLIP-Driven-Universal-Model')
if '/CLIP-Driven-Universal-Model' not in sys.path:
    sys.path.append('/CLIP-Driven-Universal-Model')
from model.Universal_model import Universal_model
from dataset.dataloader import get_loader
from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from utils.utils import get_key, dice_score, TEMPLATE, ORGAN_NAME
from tqdm import tqdm
import numpy as np

model_checkpoint = "output/unet/epoch_50.pth"
# CUDA_VISIBLE_DEVICES=0 python -W ignore -m torch.distributed.launch --nproc_per_node=1 --master_port=1234 train.py --dist True --data_root_path /data/clip/ClipReady/ --num_workers 10 --num_samples 4 --cache_dataset --cache_rate 0.3 --uniform_sample --dataset_list PAOT_TEMP --datasetkey 01 04 05 07 08 09 10_03 10_06 10_07 10_08 10_09 10_10
back_bone = log_name = 'unet'
roi_x = roi_y = roi_z = 96
trans_encoding = "word_embedding"
word_embedding = "./pretrained_weights/txt_encoding.pth"
from utils import loss
NUM_CLASS = 32

import os
import torch
def wrap_around(tensor, shift):
    n = tensor.shape[0]
    idx = [(i + shift) % n for i in range(n)]
    return tensor[idx]
def create_model(backbone, num_classes, trans_encoding, epoch, log_name, result_path="/data/result", word_embedding = "./pretrained_weights/txt_encoding.pth" ,model_pretrain=None, model_checkpoint= None, roi_x=96, roi_y=96, roi_z=96, swap_word_embedding=0,device = 0):
    # Construct the default model path if not provided
    if model_checkpoint is None:
        model_checkpoint = os.path.join(result_path, log_name, f"epoch_{epoch}.pth")

    # Initialize the model
    model = Universal_model(
        img_size=(roi_x, roi_y, roi_z),
        in_channels=1,
        out_channels=num_classes,
        backbone=backbone,
        encoding=trans_encoding
    )

    # Load pre-trained weights if provided
    if model_pretrain is not None:
        if model_pretrain is not None and os.path.exists(model_pretrain):
            model.load_params(torch.load(model_pretrain)["state_dict"])
            print(f"Loaded pre-trained model from {model_pretrain}")
        else:
            print(f"No pre-trained model found at {model_pretrain}, initializing model from scratch")

    # Load word embedding if encoding is 'word_embedding'
    if trans_encoding == 'word_embedding':
        word_embedding = torch.load(word_embedding)
        if swap_word_embedding != -1:
            word_embedding = wrap_around(word_embedding, swap_word_embedding)
        model.organ_embedding.data = word_embedding.float()
        print('Loaded word embedding')

    # Log the model creation
    log_dir = os.path.join(result_path, log_name)
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, "model_creation_log.txt")

    with open(log_file_path, "a") as log_file:
        log_file.write(f"Model created with parameters: backbone={backbone}, num_classes={num_classes}, trans_encoding={trans_encoding}, epoch={epoch}, roi=({roi_x}, {roi_y}, {roi_z})\n")
    model.to(f"cuda:{device}")
    return model

# model = create_model(back_bone,NUM_CLASS, trans_encoding, 50, log_name, model_checkpoint=model_checkpoint)

def train(args, train_loader, model, optimizer, loss_seg_DICE, loss_seg_CE):
    model.train()
    # change to validate mode
    args.epoch=0
    loss_bce_ave = 0
    loss_dice_ave = 0
    epoch_iterator = tqdm(
        train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
    )
    for step, batch in enumerate(epoch_iterator):
        x, y, name = batch["image"].to(args.device), batch["post_label"].float().to(args.device), batch['name']
        logit_map = model(x)
        term_seg_Dice = loss_seg_DICE.forward(logit_map, y, name, TEMPLATE)
        term_seg_BCE = loss_seg_CE.forward(logit_map, y, name, TEMPLATE)
        loss = term_seg_BCE + term_seg_Dice
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        epoch_iterator.set_description(
            "Epoch=%d: Training (%d / %d Steps) (dice_loss=%2.5f, bce_loss=%2.5f)" % (
                args.epoch, step, len(train_loader), term_seg_Dice.item(), term_seg_BCE.item())
        )
        loss_bce_ave += term_seg_BCE.item()
        loss_dice_ave += term_seg_Dice.item()
        torch.cuda.empty_cache()
    print('Epoch=%d: ave_dice_loss=%2.5f, ave_bce_loss=%2.5f' % (args.epoch, loss_dice_ave/len(epoch_iterator), loss_bce_ave/len(epoch_iterator)))
    
    return loss_dice_ave/len(epoch_iterator), loss_bce_ave/len(epoch_iterator)
def validation_patch(model, ValLoader, args):
    model.eval()

    output_dir = '/data/clip/result/unet/prediction'
    os.makedirs(output_dir, exist_ok=True)

    roi_size = (args.roi_x, args.roi_y, args.roi_z)
    overlap = 0.25

    for index, batch in enumerate(tqdm(ValLoader)):
        image, label, name = batch["image"].to(args.device), batch["post_label"], batch["name"]
        print(name, image.shape)
        template_key = get_key(name[0])
        prediction_data = {
            "preds": [],
            "labels": [],
            "xs": [],
            "weights": [],
            "biases": [],
            "coords": []
        }

        image_splitter = SlidingWindowSplitter(patch_size=roi_size, overlap=overlap)
        label_splitter = SlidingWindowSplitter(patch_size=roi_size, overlap=overlap)

        def predictor(input_data):
            output, x_feat, xs, weights, biases = model(input_data, True)
            return output, xs, weights, biases

        with torch.no_grad():
            image_patches = list(image_splitter(image))
            label_patches = list(label_splitter(label))

            for (patch_data, coord), (label_data, _) in zip(image_patches, label_patches):
                patch_data = patch_data.to(args.device)
                label_data = label_data.to(args.device)
                pred, xs, weights, biases = predictor(patch_data)

                pred_sigmoid = F.sigmoid(pred)

                prediction_data["preds"].append(pred_sigmoid.cpu())
                prediction_data["labels"].append(label_data.cpu())
                prediction_data["xs"].append(xs)
                prediction_data["weights"].append(weights)
                prediction_data["biases"].append(biases)
                prediction_data["coords"].append(coord)
        path = os.path.join(output_dir, f"{name[0]}.pt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(prediction_data, path)

    if args.local_rank == 0:
        with open('out/' + args.log_name + f'/val_{args.epoch}.txt', 'w') as f:
            for key in TEMPLATE.keys():
                organ_list = TEMPLATE[key]
                content = 'Task%s| ' % (key)
                for organ in organ_list:
                    content += '%s: saved, ' % (ORGAN_NAME[organ-1])
                print(content)
                f.write(content)
                f.write('\n')

def validation(model, ValLoader, args):
    model.eval()
    dice_list = {}
    individual_data = {}
    for key in TEMPLATE.keys():
        dice_list[key] = np.zeros((2, NUM_CLASS)) # 1st row for dice, 2nd row for count
        keys = ["preds", "labels", "weights", "biases"]
        individual_data[key] = {k:[[] for _ in range(NUM_CLASS)] for k in keys}
        individual_data[key]["x_feature"] = []

    for index, batch in enumerate(tqdm(ValLoader)):
        # print('%d processd' % (index))
        image, label, name = batch["image"].to(args.device), batch["post_label"], batch["name"]
        print(name, image.shape)
        with torch.no_grad():
            pred, list_list_x_feat, list_list_xs, list_list_weights,list_list_biases = sliding_window_inference_with_meta_data(image, (args.roi_x, args.roi_y, args.roi_z), 1, model)
            pred_sigmoid = F.sigmoid(pred)
        
        B = pred_sigmoid.shape[0]    
        for b in range(B):
            template_key = get_key(name[b])
            organ_list = TEMPLATE[template_key]
            individual_data[template_key]["x_feature"].append(list_list_x_feat)
            for organ in organ_list:
                dice_organ = dice_score(pred_sigmoid[b,organ-1,:,:,:], label[b,organ-1,:,:,:].cuda())
                dice_list[template_key][0][organ-1] += dice_organ[0]
                dice_list[template_key][1][organ-1] += 1
                individual_data[template_key]["preds"][organ-1].append(pred_sigmoid[b,organ-1,:,:,:])
                individual_data[template_key]["labels"][organ-1].append(label[b,organ-1,:,:,:])
                individual_data[template_key]["biases"][organ-1].append([list_biases[b] for list_biases in list_list_biases])
                individual_data[template_key]["weights"][organ-1].append([list_weights[b] for list_weights in list_list_weights])
            

    
    ave_organ_dice = np.zeros((2, NUM_CLASS))
    if args.local_rank == 0:
        with open('out/'+args.log_name+f'/val_{args.epoch}.txt', 'w') as f:
            for key in TEMPLATE.keys():
                organ_list = TEMPLATE[key]
                content = 'Task%s| '%(key)
                for organ in organ_list:
                    dice = dice_list[key][0][organ-1] / dice_list[key][1][organ-1]
                    content += '%s: %.4f, '%(ORGAN_NAME[organ-1], dice)
                    ave_organ_dice[0][organ-1] += dice_list[key][0][organ-1]
                    ave_organ_dice[1][organ-1] += dice_list[key][1][organ-1]
                print(content)
                f.write(content)
                f.write('\n')
            content = 'Average | '
            for i in range(NUM_CLASS):
                content += '%s: %.4f, '%(ORGAN_NAME[i], ave_organ_dice[0][organ-1] / ave_organ_dice[1][organ-1])
            print(content)
            f.write(content)
            f.write('\n')
def sliding_window_inference_with_meta_data(image, roi_size, sw_batch_size, model):
    list_x_feat, list_xs, list_weights, list_biases = [], [], [], []
    count = 0
    def wrap_predictor(input_data):
        ouput, x_feat, xs, weights,biases = model(input_data, True)
        nonlocal count
        count += 1
        list_x_feat.append(x_feat)
        list_xs.append(xs)
        list_weights.append(weights)
        list_biases.append(biases)
        return ouput
    total_output = sliding_window_inference(image, roi_size, sw_batch_size, wrap_predictor)
    return total_output, list_x_feat, list_xs, list_weights, list_biases
    


def valid_loop(data_root_path, dataset_list, datasetkey, num_workers, num_samples, cache_dataset, cache_rate, batch_size, roi, space,device=0):
    class _object:
        pass
    
    args = _object()
    args.data_root_path = data_root_path
    args.dataset_list = dataset_list
    args.datasetkey = datasetkey
    args.num_workers = num_workers
    args.num_samples = num_samples
    args.cache_dataset = cache_dataset
    args.cache_rate = cache_rate
    args.uniform_sample = True
    args.batch_size = batch_size
    args.roi_x = args.roi_y = args.roi_z = roi
    args.space_x = args.space_y = args.space_z = space
    args.a_min, args.a_max, args.b_min, args.b_max = -175,250,0.0,1.0
    args.data_txt_path = './dataset/dataset_list/'
    args.phase = "validation"
    args.encoding = "word_embedding"
    args.dist = False
    args.lr = 1e-4
    args.weight_decay = 1e-5
    args.device = torch.device(f"cuda:{device}")
    args.backbone = "unet"
    args.max_epoch = 50
    args.warmup_epoch = 5
    args.log_name = "unet"
    args.result_path = "/data/result"
    args.word_embedding = "./pretrained_weights/txt_encoding.pth"
    args.model_checkpoint = "output/unet/epoch_50.pth"
    args.swap_word_embedding = 0
    args.num_workers = 0
    args.dist = False
    args.device = 0

    model = create_model(args.backbone, NUM_CLASS,trans_encoding=args.encoding, epoch=50, log_name=args.log_name, result_path=args.result_path, word_embedding=args.word_embedding, model_pretrain=None, model_checkpoint=args.model_checkpoint, roi_x=args.roi_x, roi_y=args.roi_y, roi_z=args.roi_z, swap_word_embedding=args.swap_word_embedding,device = args.device)
    loss_seg_DICE = loss.DiceLoss(num_classes=NUM_CLASS).to(args.device)
    loss_seg_CE = loss.Multi_BCELoss(num_classes=NUM_CLASS).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=args.warmup_epoch, max_epochs=args.max_epoch)
    
    val_loader, val_transform = get_loader(args)
    if args.phase == "train":
        train(args, val_loader, model, optimizer, loss_seg_DICE, loss_seg_CE)
    else:
        validation_patch(model, val_loader, args)
    # validation(model, val_loader, args)

valid_loop('/data/clip/ClipReady/', ['PAOT_TEMP'], '01 04 05 07 08 09 10_03 10_06 10_07 10_08 10_09 10_10'.split(), 10, 2, False, 0.3, 1, 96, 1.5, 0)