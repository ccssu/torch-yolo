
    # ---------------------------------------------------------------#
    import torch
    weights = '/home/fengwen/weights/yolov5s.pt'
    ckpt = torch.load(weights, map_location='cpu')
    model = Model(cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
    exclude = ['anchor'] if (cfg or hyp.get('anchors')) and not resume else []  # exclude keys
    new_parameters = dict()
    for key, value in ckpt['model'].state_dict().items():
        if 'num_batches_tracked' in key:
            continue
        value = oneflow.tensor(value.detach().cpu().numpy(),dtype=oneflow.float32).float()

        new_parameters[key] = value

    csd = intersect_dicts(new_parameters, model.state_dict(), exclude=exclude)  # intersect
    model.load_state_dict(csd, strict=False)  # load
    pretrained = True

    # ---------------------------------------------------------------------------------------------------#
 