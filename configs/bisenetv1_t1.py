cfg = dict(
    model_type='bisenetv1',
    n_cats=2,
    num_aux_heads=2,
    lr_start=1e-2,
    weight_decay=1e-4,
    warmup_iters=1000,
    max_iter=254200,
    dataset='T1',
    im_root='./datasets/t1',
    train_im_anns='./datasets/t1/train.txt',
    val_im_anns='',
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
