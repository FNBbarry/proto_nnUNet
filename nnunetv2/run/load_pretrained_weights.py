import torch
from torch._dynamo import OptimizedModule
from torch.nn.parallel import DistributedDataParallel as DDP


def load_pretrained_weights(network, fname, verbose=False):
    """
    Transfers all weights between matching keys in state_dicts. matching is done by name and we only transfer if the
    shape is also the same. Segmentation layers (the 1x1(x1) layers that produce the segmentation maps)
    identified by keys ending with '.seg_layers') are not transferred!

    If the pretrained weights were optained with a training outside nnU-Net and DDP or torch.optimize was used,
    you need to change the keys of the pretrained state_dict. DDP adds a 'module.' prefix and torch.optim adds
    '_orig_mod'. You DO NOT need to worry about this if pretraining was done with nnU-Net as
    nnUNetTrainer.save_checkpoint takes care of that!

    """
    saved_model = torch.load(fname)
    # pretrained_dict = saved_model['network_weights']
    pretrained_dict = saved_model['model_state_dict']

    skip_strings_in_pretrained = [
        '.prompt_encoder.',
        '.mask_decoder.'
    ]

    if isinstance(network, DDP):
        mod = network.module
    else:
        mod = network
    if isinstance(mod, OptimizedModule):
        mod = mod._orig_mod

    model_dict = mod.state_dict()
    # verify that all but the segmentation layers have the same shape
    for key, _ in model_dict.items():
        if 'backbone' in key:# for other pretrained pth
            if all([i not in key for i in skip_strings_in_pretrained]):
                if key == 'nlp.weight' or key == 'nlp.bias' or key=='prototypes':
                    continue
                # [9:]for other pretrained pth
                assert key[9:] in pretrained_dict, \
                    f"Key {key} is missing in the pretrained model weights. The pretrained weights do not seem to be " \
                    f"compatible with your network."
                # if key == 'prototypes' or key== 'backbone.outc.conv.weight' or  key== 'backbone.outc.conv.bias' or key=='mask_norm.weight' or key=='mask_norm.bias':
                #     continue
                assert model_dict[key].shape == pretrained_dict[key[9:]].shape, \
                    f"The shape of the parameters of key {key} is not the same. Pretrained model: " \
                    f"{pretrained_dict[key[9:]].shape}; your network: {model_dict[key]}. The pretrained model " \
                    f"does not seem to be compatible with your network."

    # fun fact: in principle this allows loading from parameters that do not cover the entire network. For example pretrained
    # encoders. Not supported by this function though (see assertions above)

    # commenting out this abomination of a dict comprehension for preservation in the archives of 'what not to do'
    # pretrained_dict = {'module.' + k if is_ddp else k: v
    #                    for k, v in pretrained_dict.items()
    #                    if (('module.' + k if is_ddp else k) in model_dict) and
    #                    all([i not in k for i in skip_strings_in_pretrained])}

    # pretrained_dict = {k: v for k, v in pretrained_dict.items()
    #                    if k in model_dict.keys() and all([i not in k for i in skip_strings_in_pretrained])}
    pretrained_dict_tmp = {}
    for k,v in pretrained_dict.items():
        k = 'backbone.'+k# for other pretrained pth
        if k in model_dict.keys() and all([i not in k for i in skip_strings_in_pretrained]):
            # 当user想把已经在已知类别的数据集上训练好的模型，
            # 再在类别不一样(但和之前的类别有重叠)的数据集上重新训练一遍，
            # 可在此处直接对prototype进行修改
            # if k == 'prototypes':
            #     continue
            if k == 'backbone.outc.conv.weight' or k == 'backbone.outc.conv.bias' or k == 'mask_norm.weight' or k =='mask_norm.bias':
                continue
            if k == 'prototypes' and model_dict[k].shape != pretrained_dict[k].shape:
                tmp = torch.zeros(mod.num_classes,v.shape[1],v.shape[2])
                label_label = {1:1,2:4,3:3,4:2,5:8,10:6,11:7,12:5,13:9}
                for class_ori,class_dest in label_label.items():
                    tmp[class_dest,:,:] = v[class_ori,:,:]
                device = v.device
                tmp.to(device)
                v = tmp
            pretrained_dict_tmp[k] = v
    pretrained_dict = pretrained_dict_tmp
    model_dict.update(pretrained_dict)

    print("################### Loading pretrained weights from file ", fname, '###################')
    if verbose:
        print("Below is the list of overlapping blocks in pretrained model and nnUNet architecture:")
        for key, value in pretrained_dict.items():
            print(key, 'shape', value.shape)
        print("################### Done ###################")
    mod.load_state_dict(model_dict)
    for name, param in mod.named_parameters():
        if 'image_encoder' in name:
            param.requires_grad = False