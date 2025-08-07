import ml_collections


def d(**kwargs):
    """Helper of creating a config dict."""
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()

    config.seed = 1234
    config.pred = 'noise_pred'
    config.z_shape = (4, 32, 32)

    config.autoencoder = d(
        pretrained_path='assets/stable-diffusion/autoencoder_kl_ema.pth'
    )

    config.train = d(
        n_steps=500000,
        batch_size=32,
        mode='cond',
        log_interval=10,
        eval_interval=500000,
        save_interval=100000,
    )

    config.optimizer = d(
        name='adamw',
        lr=0.00002,
        weight_decay=0.03,
        betas=(0.99, 0.99),
    )

    config.lr_scheduler = d(
        name='customized',
        warmup_steps=2500
    )

    config.nnet = d(
        name='uvit',
        img_size=32,
        patch_size=2,
        in_chans=4,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=False,
        mlp_time_embed=False,
        num_classes=16,
        use_checkpoint=True,
        conv=False,
        use_dyt="DyTWithFFT",
    )

    config.dataset = d(
        name='custom',  # 自定义数据集（需代码支持）
        path=r'assets/datasets/wool',
        resolution=256,      # 与模型输入尺寸一致
        random_flip=True,    # 启用随机水平翻转
        cfg=True,
        p_uncond=0.1
    )

    config.sample = d(
        sample_steps=50,
        n_samples=380,
        mini_batch_size=5,  # the decoder is large
        algorithm='dpm_solver',
        cfg=True,
        scale=0.4,
        path=''
    )

    return config
