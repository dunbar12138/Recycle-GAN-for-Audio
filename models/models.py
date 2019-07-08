
def create_model(opt):
    model = None
    print(opt.model)
    if opt.model == 'cycle_gan':
        assert(opt.dataset_mode == 'unaligned' or opt.dataset_mode == 'spectrogram')
        from .cycle_gan_model import CycleGANModel
        model = CycleGANModel()
    elif opt.model == 'recycle_gan':
        assert(opt.dataset_mode == 'unaligned_triplet' or opt.dataset_mode == 'triplet_spectrogram')
        from .recycle_gan_model import RecycleGANModel
        model = RecycleGANModel()
    elif opt.model == 'reCycle_gan':
        assert(opt.dataset_mode == 'unaligned_triplet')
        from .reCycle_gan_model import ReCycleGANModel
        model = ReCycleGANModel()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
