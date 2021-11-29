cfg = dict(
    model_type='bisenetv2',
    n_cats=2,
    num_aux_heads=4,
    lr_start=5e-3,
    weight_decay=1e-4,
    warmup_iters=1000,
    max_iter=57450,
    dataset='T1',
    im_root='./datasets/t1',
    train_im_anns='./datasets/t1/train256.txt',
    val_im_anns='./datasets/t1/val256.txt',
    scales=[0.5, 2.],
    cropsize=[256, 256],
    eval_crop=[256, 256],
    eval_scales=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
    ims_per_gpu=16,
    eval_ims_per_gpu=1,
    use_fp16=False,
    use_sync_bn=False,
    respth='/content/gdrive/MyDrive/Università/IMT_Thesis/Datasets/BraTS2019/BiSeNet',
)
