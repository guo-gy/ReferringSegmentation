from .segmenter import CRIS
from .cris_p import CRIS_P, build_cris_p
from .cris_lite import CRIS_Lite, build_cris_lite
from loguru import logger

# def build_segmenter(args):
#     model = CRIS(args)
#     backbone = []
#     backbone_no_decay = []
#     head = []
#     for k, v in model.named_parameters():
#         if k.startswith('backbone') and 'positional_embedding' not in k:
#             backbone.append(v)
#         elif 'positional_embedding' in k:
#             backbone_no_decay.append(v)
#         else:
#             head.append(v)
#     print('Backbone with decay: {}, Backbone without decay: {}, Head: {}'.format(
#         len(backbone), len(backbone_no_decay), len(head)))
#     param_list = [{
#         'params': backbone,
#         'initial_lr': args.lr_multi * args.base_lr
#     }, {
#         'params': backbone_no_decay,
#         'initial_lr': args.lr_multi * args.base_lr,
#         'weight_decay': 0
#     }, {
#         'params': head,
#         'initial_lr': args.base_lr
#     }]
#     return model, param_list


def build_segmenter(args):
    # 根据配置选择模型
    if hasattr(args, 'model_name') and args.model_name == 'cris_p':
        model = CRIS_P(args)
        logger.info('Using CRIS-P with Cross-Modal Alignment')
    elif hasattr(args, 'model_name') and args.model_name == 'cris_lite':
        model = CRIS_Lite(args)
        logger.info('Using CRIS-Lite (lightweight for testing)')
    else:
        model = CRIS(args)
        logger.info('Using standard CRIS')

    backbone = []
    head = []
    for k, v in model.named_parameters():
        if k.startswith('backbone') and 'positional_embedding' not in k:
            backbone.append(v)
        else:
            head.append(v)
    logger.info('Backbone with decay={}, Head={}'.format(len(backbone), len(head)))
    param_list = [{
        'params': backbone,
        'initial_lr': args.lr_multi * args.base_lr
    }, {
        'params': head,
        'initial_lr': args.base_lr
    }]
    return model, param_list
